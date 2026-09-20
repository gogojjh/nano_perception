#!/usr/bin/env python3
"""NanoOWL+NanoSAM 分割 ROS 桥接节点（系统 python3.8 / ROS1 noetic）。

订阅 /odin1/image/undistorted（bgr8），按 max_rate 节流（默认 5 Hz）、忙时丢帧，
把 JPEG 发给本地推理引擎（nano_perception_engine.py），发布：
  /nano_perception/segmentation_image  彩色叠加图（bgr8，掩码+框+分数）
  /nano_perception/segmentation_result JSON 摘要（std_msgs/String，每类的 score/box_xyxy）
  /nano_perception/masks               mono16 位掩码图（第 i 个类别占 bit i，0=背景）
  /nano_perception/engine_latency      引擎单帧耗时 ms（std_msgs/Float64）

用法：python3 nano_perception_bridge.py _engine_port:=8891 _prompts:="person, chair"
"""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String

DEFAULT_PROMPTS = (
    "fire hydrant, toilet, refrigerator, trash can, potted plant, wheelchair"
)

# 每类固定 BGR 颜色（按 prompts 顺序取），超出循环取色
PALETTE_BGR = [
    (60, 20, 220),  # fire hydrant  红
    (60, 180, 75),  # toilet        绿
    (255, 225, 25),  # refrigerator  黄
    (200, 20, 60),  # trash can     紫
    (100, 100, 100),  # potted plant  灰
    (255, 255, 255),  # wheelchair    白
]


class NanoPerceptionBridge(object):
    def __init__(self):
        prompts_raw = rospy.get_param("~prompts", DEFAULT_PROMPTS)
        if isinstance(prompts_raw, list):
            self.prompts = [str(p).strip() for p in prompts_raw if str(p).strip()]
        else:
            self.prompts = [p.strip() for p in str(prompts_raw).split(",") if p.strip()]
        self.confidence_threshold = float(
            rospy.get_param("~confidence_threshold", 0.1)
        )
        self.max_rate = float(rospy.get_param("~max_rate", 5.0))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))
        engine_port = int(rospy.get_param("~engine_port", 8891))
        self.engine_url = "http://127.0.0.1:{}/infer".format(engine_port)

        # 存盘开关：save_dir 非空 = 每成功帧存 overlay 图 + JSON 摘要（默认关闭）
        self.save_dir = str(rospy.get_param("~save_dir", "")).strip()
        self.save_max_frames = int(rospy.get_param("~save_max_frames", 0))
        # 额外存原始相机帧（raw_<stamp>.jpg），供离线阈值扫描等复用（默认关闭）
        self.save_raw_frames = int(rospy.get_param("~save_raw_frames", 0)) > 0
        self.saved_count = 0
        if self.save_dir:
            try:
                os.makedirs(self.save_dir, exist_ok=True)
            except Exception as exc:
                rospy.logwarn("存盘目录创建失败，本次运行不存盘: %s", exc)
                self.save_dir = ""

        self.bridge = CvBridge()
        # 流水线：最多 2 帧在飞（1 帧在引擎、1 帧在编码），忙时丢帧。
        # 第 2 帧的 JPEG 编码与第 1 帧的引擎推理重叠，吞吐逼近引擎上限。
        self.in_flight = 0
        self.in_flight_max = 2
        self.in_flight_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="np_proc")
        # 引擎串行处理：记录预测的空闲时刻（monotonic 秒），后续请求等到此时再发
        self.engine_free_at = 0.0
        self.engine_lock = threading.Lock()
        self.last_pub = rospy.Time(0)
        rospy.Subscriber(
            "/odin1/image/undistorted", Image, self.on_image, queue_size=1
        )
        self.pub_img = rospy.Publisher(
            "/nano_perception/segmentation_image", Image, queue_size=1
        )
        self.pub_result = rospy.Publisher(
            "/nano_perception/segmentation_result", String, queue_size=1
        )
        self.pub_masks = rospy.Publisher(
            "/nano_perception/masks", Image, queue_size=1
        )
        self.pub_latency = rospy.Publisher(
            "/nano_perception/engine_latency", Float64, queue_size=1
        )
        rospy.loginfo(
            "nano_perception_bridge 就绪: prompts=%s, max_rate=%.2f Hz, "
            "engine=%s, save_dir=%s",
            self.prompts,
            self.max_rate,
            self.engine_url,
            self.save_dir or "(不存盘)",
        )
        if self.save_dir and self.save_raw_frames:
            rospy.loginfo("原始帧存盘已开启（raw_*.jpg）")

    def on_image(self, msg):
        """节流 + 丢帧：忙（在飞帧数满）或距上次提交不足一个周期就丢弃，绝不排队。"""
        with self.in_flight_lock:
            if self.in_flight >= self.in_flight_max:
                return
            if (rospy.Time.now() - self.last_pub).to_sec() < 1.0 / self.max_rate:
                return
            self.in_flight += 1
            self.last_pub = rospy.Time.now()
        self.pool.submit(self._process_frame, msg)

    def _process_frame(self, msg):
        """工作线程：转换 -> JPEG -> 请求引擎 -> 画图 -> 发布。"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_img = np.ascontiguousarray(cv_img)
            ok, buf = cv2.imencode(
                ".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                rospy.logwarn_throttle(10, "JPEG 编码失败，丢帧")
                return
            data = self.request_engine(buf)
            if data is not None:
                self.publish_results(msg, cv_img, data)
        except Exception as exc:
            rospy.logwarn_throttle(10, "帧处理异常: %s", exc)
        finally:
            with self.in_flight_lock:
                self.in_flight -= 1

    def request_engine(self, jpeg_buf):
        """把 JPEG 发给引擎，返回解析后的 JSON dict；忙/失败返回 None。"""
        payload = json.dumps(
            {
                "image_b64": base64.b64encode(jpeg_buf.tobytes()).decode("ascii"),
                "prompts": self.prompts,
                "confidence_threshold": self.confidence_threshold,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.engine_url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            # 先等到预测的引擎空闲时刻（上一帧占用到 t_send+server_ms），
            # 流水线的第 2 帧就不用靠 503 重试碰运气。
            with self.engine_lock:
                wait_s = self.engine_free_at - time.monotonic()
            if wait_s > 0:
                time.sleep(min(wait_s, 0.3))
            # 兜底：预测不准时引擎忙（503）是瞬时的，短等重试即可接上
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        with self.engine_lock:
                            # 响应到达 = 引擎已空闲，下一帧等到此时+5ms 余量再发
                            self.engine_free_at = max(
                                self.engine_free_at, time.monotonic() + 0.005
                            )
                        return data
                except urllib.error.HTTPError as exc:
                    if exc.code == 503 and attempt < 4:
                        time.sleep(0.02)
                        continue
                    if exc.code == 503:
                        rospy.logwarn_throttle(10, "引擎忙（503），丢帧")
                    else:
                        rospy.logwarn_throttle(10, "引擎错误 HTTP %s", exc.code)
                    return None
        except Exception as exc:  # 超时/连接失败，引擎重启后自动恢复
            rospy.logwarn_throttle(10, "引擎请求失败: %s", exc)
        return None

    def publish_results(self, src_msg, cv_img, data):
        overlay = self.draw_overlay(cv_img, data["results"])
        # 懒发布：publisher 没有订阅者就跳过消息构造与发布（每帧实时查询，
        # 订阅者后连下一帧自动恢复）。存盘逻辑不受影响。
        if self.pub_img.get_num_connections() > 0:
            out_img = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            out_img.header = src_msg.header
            self.pub_img.publish(out_img)

        summary = None
        if self.pub_result.get_num_connections() > 0 or self.save_dir:
            summary = {
                "stamp_ns": src_msg.header.stamp.to_nsec(),
                "engine_ms": round(float(data.get("engine_ms", 0.0)), 1),
                "detections": [
                    {
                        "label": r["prompt"],
                        "score": float(s),
                        "box_xyxy": [float(v) for v in b],
                    }
                    for r in data["results"]
                    for s, b in zip(r["scores"], r["boxes"])
                ],
            }
            if self.pub_result.get_num_connections() > 0:
                self.pub_result.publish(json.dumps(summary, ensure_ascii=False))

        if self.pub_masks.get_num_connections() > 0:
            mask_map = self.build_mask_map(data["results"], cv_img.shape[:2])
            if mask_map is not None:
                self.pub_masks.publish(self.mask_map_to_msg(mask_map, src_msg))

        if self.pub_latency.get_num_connections() > 0:
            self.pub_latency.publish(Float64(float(data.get("engine_ms", 0.0))))

        self.save_results(overlay, summary, cv_img)

    def save_results(self, overlay, summary, raw_img=None):
        """把 overlay 图 + JSON 摘要（可选原始帧）落到 save_dir；失败降级不影响主流程。"""
        if not self.save_dir:
            return
        if self.save_max_frames and self.saved_count >= self.save_max_frames:
            rospy.logwarn_once(
                "已达存盘帧数上限 %d，停止存盘", self.save_max_frames
            )
            return
        try:
            # 文件名用相机消息时间戳，与发布 JSON 的 stamp_ns 一一对应
            stamp_ns = int(summary.get("stamp_ns", 0))
            if stamp_ns <= 0:  # 无时间戳的帧回退壁钟，避免文件名冲突
                stamp_ns = rospy.Time.now().to_nsec()
            name = "{}.{:09d}".format(stamp_ns // 10**9, stamp_ns % 10**9)
            jpg_path = os.path.join(self.save_dir, name + ".jpg")
            json_path = os.path.join(self.save_dir, name + ".json")
            ok, buf = cv2.imencode(
                ".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if ok:
                with open(jpg_path, "wb") as f:
                    f.write(buf.tobytes())
            else:
                rospy.logwarn_throttle(10, "overlay 编码失败，跳过存图")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False)
            if self.save_raw_frames and raw_img is not None:
                ok, buf = cv2.imencode(
                    ".jpg", raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )
                if ok:
                    with open(
                        os.path.join(self.save_dir, "raw_" + name + ".jpg"), "wb"
                    ) as f:
                        f.write(buf.tobytes())
                else:
                    rospy.logwarn_throttle(10, "原始帧编码失败，跳过存图")
            self.saved_count += 1
        except Exception as exc:  # 磁盘满/权限等，跳过不拖垮发布
            rospy.logwarn_throttle(10, "存盘失败（跳过，不影响主流程）: %s", exc)

    def draw_overlay(self, img, results):
        """半透明掩码染色 + 框 + 'label score' 文字。

        掩码先在引擎输出分辨率合并成类别索引图，一次 NEAREST 放大到原图尺寸
        再用 cv2.LUT 查表染色，避免逐掩码全分辨率 resize/布尔染色的高开销。
        """
        vis = img.copy()
        idx_map = None
        for i, r in enumerate(results):
            for mask_b64 in r["masks_png_b64"]:
                mask = self.decode_mask(mask_b64)
                if mask is None:
                    continue
                if idx_map is None:
                    idx_map = np.zeros(mask.shape, dtype=np.uint8)
                if mask.shape != idx_map.shape:
                    mask = (
                        cv2.resize(
                            mask.astype(np.uint8),
                            (idx_map.shape[1], idx_map.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 0
                    )
                idx_map[mask] = i + 1
        if idx_map is not None:
            idx_full = cv2.resize(
                idx_map,
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            # 类别索引 numpy 查表染色（背景 0 用黑色占位，fg 掩码限定不染背景）
            pal_full = np.zeros((len(PALETTE_BGR) + 1, 3), dtype=np.uint8)
            pal_full[1:] = PALETTE_BGR
            color_full = pal_full[idx_full]
            fg = idx_full > 0
            vis[fg] = (vis[fg] * 0.45 + color_full[fg] * 0.55).astype(np.uint8)
        for i, r in enumerate(results):
            color = PALETTE_BGR[i % len(PALETTE_BGR)]
            for box, score, _ in zip(r["boxes"], r["scores"], r["masks_png_b64"]):
                x0, y0, x1, y1 = (int(v) for v in box)
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    vis,
                    "{} {:.2f}".format(r["prompt"], score),
                    (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        return vis

    def build_mask_map(self, results, shape):
        """把各类掩码并成 mono16 位掩码图：像素值 = Σ 2^i（第 i 类命中）。

        先在引擎输出分辨率合并位图（NEAREST 放大不改变位模式），
        再一次放大到目标尺寸，避免逐掩码全分辨率 resize。
        """
        mask_map = None
        for i, r in enumerate(results):
            for mask_b64 in r["masks_png_b64"]:
                mask = self.decode_mask(mask_b64)
                if mask is None:
                    continue
                if mask_map is None:
                    mask_map = np.zeros(mask.shape, dtype=np.uint16)
                if mask.shape != mask_map.shape:
                    mask = (
                        cv2.resize(
                            mask.astype(np.uint8),
                            (mask_map.shape[1], mask_map.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 0
                    )
                mask_map[mask] |= np.uint16(1 << i)
        if mask_map is None:
            return None
        if mask_map.shape != shape:
            mask_map = cv2.resize(
                mask_map, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return mask_map

    @staticmethod
    def mask_map_to_msg(mask_map, src_msg):
        msg = Image()
        msg.header = src_msg.header
        msg.height, msg.width = mask_map.shape
        msg.encoding = "mono16"
        msg.is_bigendian = 0
        msg.step = mask_map.shape[1] * 2
        msg.data = mask_map.tobytes()
        return msg

    @staticmethod
    def decode_mask(b64):
        """base64 掩码 -> bool 数组。

        'rle:' 前缀 = 游程编码（uint32 小端 [h, w, run0, run1, ...]，run 从 0 值开始交替）；
        否则按旧 PNG 格式解码（兼容旧引擎）。
        """
        if b64.startswith("rle:"):
            raw = base64.b64decode(b64[4:])
            runs = np.frombuffer(raw, dtype="<u4")
            # 合法结构：[h, w] + 至少 1 段；段数奇偶不定（掩码结尾为背景时尾段
            # 是 0 值段、段数为奇数），只能靠像素总和校验。
            h, w = int(runs[0]), int(runs[1]) if len(runs) >= 3 else (0, 0)
            if h <= 0 or w <= 0 or int(np.sum(runs[2:])) != h * w:
                rospy.logwarn_throttle(10, "RLE 掩码数据损坏，丢弃该掩码")
                return None
            # 向量化展开：段值 0/255 交替（段 0 从 0 开始），np.repeat 按段长复制
            vals = np.zeros(len(runs) - 2, dtype=np.uint8)
            vals[1::2] = 255
            flat = np.repeat(vals, runs[2:].astype(np.int64))
            return flat.reshape(h, w) > 127
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE) > 127


def main():
    rospy.init_node("nano_perception_bridge")
    NanoPerceptionBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
