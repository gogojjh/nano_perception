#!/usr/bin/env python3
"""NanoOWL + NanoSAM 推理引擎：本地 HTTP 服务，供 ROS 桥接节点调用。

运行环境：conda env nano_perception（python 3.8 + NVIDIA JP5 版 torch 2.1.0 nv23.06）
流程：OWL 开放词表检测（TensorRT 图像编码器）→ 每个检测框用 NanoSAM 分割（两个 TRT engine）
接口：
  GET  /health -> {"status": "ready"}（预热完成后才 ready）
  POST /infer  body: {"image_b64": "<jpeg base64>", "prompts": [...],
                      "confidence_threshold": 0.1}
               -> {"engine_ms": ..., "detect_ms": ..., "sam_encode_ms": ...,
                   "sam_decode_ms": ...,
                   "results": [{"prompt", "scores", "boxes", "masks_png_b64"}]}
               忙时返回 503 {"error": "busy"}（桥接端靠它丢帧）
掩码为二值图 base64 编码：MASK_CODEC 环境变量选格式（fast=torchvision PNG 默认 /
rle=游程 / pil=旧 PIL PNG），boxes 为原图尺寸 XYXY。
输入图像先等比缩放到高 768（1600x1296 太大），坐标与掩码统一还原回原图。
"""
from pathlib import Path

import argparse
import base64
import io
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DEFAULT_PROMPTS = (
    "fire hydrant, toilet, refrigerator, trash can, potted plant, wheelchair"
)


def _rle_encode(mask_u8: np.ndarray) -> str:
    """游程编码：'rle:' + base64(uint32 小端 [h, w, run0, run1, ...])，无损。

    实测（ARM64）：1024x1024 约 5ms、1600x1296 约 13ms——编码分辨率越低越快，
    因此 rle 路径在 SAM 输出分辨率（1024）上编码，放大交给桥端 cv2 兜底。
    """
    h, w = mask_u8.shape
    flat = (mask_u8 > 0).astype(np.uint8).ravel()
    changes = np.flatnonzero(np.diff(np.concatenate(([0], flat))))
    # 段边界 = [0] + 变化点 + [总长]，diff 后得到全部段长（含开头段，不能丢）
    runs = np.diff(np.concatenate(([0], changes, [len(flat)])))
    payload = np.concatenate((np.array([h, w], dtype="<u4"), runs.astype("<u4")))
    return "rle:" + base64.b64encode(payload.tobytes()).decode("ascii")


def encode_mask(mask_u8: np.ndarray) -> str:
    """按 MASK_CODEC 编码原图分辨率掩码：fast=torchvision PNG，pil=PIL PNG。

    rle 不经过这里（在 infer 里用 SAM 分辨率直接编码）。fast 在 ARM64 实测比
    pil 慢 40%，仅作兼容选项；torchvision 不可用时自动回退 pil。
    """
    codec = os.environ.get("MASK_CODEC", "pil")
    if codec == "fast":
        try:
            from torchvision.io import encode_png

            t = torch.from_numpy(np.ascontiguousarray(mask_u8))[None]
            return base64.b64encode(encode_png(t).numpy().tobytes()).decode("ascii")
        except Exception:
            codec = "pil"
    # PIL PNG（默认回退，MASK_CODEC=pil 或 fast 不可用时）
    buf = io.BytesIO()
    Image.fromarray(mask_u8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoOWL+NanoSAM 推理引擎")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument(
        "--repo-path", default=str(Path(__file__).resolve().parent)
    )
    parser.add_argument("--data-path", default=None, help="engine 目录，默认 <repo>/data")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS, help="逗号分隔的类别列表")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="兜底阈值：仅当 HTTP 请求未带 confidence_threshold 时生效",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=100.0,
        help="过滤小于该面积的框（resize 后图像坐标 px²），调低可召回远处小物体",
    )
    parser.add_argument("--image-height", type=int, default=768, help="输入等比缩放后的高")
    parser.add_argument(
        "--max-detections-per-prompt", type=int, default=6, help="每类最多保留的框数"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        default="nanoowl",
        choices=["nanoowl", "yoloe"],
        help="用哪套模型。nanoowl=NanoOWL 检测 + NanoSAM 分割（原有，默认）；"
             "yoloe=YOLOE 检测+分割一体（不加载 NanoSAM）。两边的 /infer 返回字段完全一致，"
             "换后端不需要动 shim / detector / rgb_overlay 任何一行",
    )
    parser.add_argument(
        "--yoloe-weights",
        default=str(Path(__file__).resolve().parent / "data/yoloe-26m-seg.pt"),
        help="--backend yoloe 时用的权重",
    )
    return parser.parse_args()


class NanoPerceptionModel:
    """持有 OwlPredictor 与 NanoSAM Predictor，并缓存 OWL 文本编码结果。"""

    def __init__(
        self,
        repo_path: str,
        data_path: str,
        device: str,
        image_height: int,
        max_detections_per_prompt: int,
        min_box_area: float = 100.0,
    ) -> None:
        from nanoowl.owl_predictor import OwlPredictor, OwlEncodeTextOutput
        from nanosam.utils.predictor import Predictor

        self.device = device
        self.image_height = image_height
        self.max_detections_per_prompt = max_detections_per_prompt
        self.min_box_area = min_box_area

        self.owl = OwlPredictor(
            model_name="google/owlvit-base-patch32",
            device=device,
            image_encoder_engine="{}/owl_image_encoder_patch32.engine".format(
                data_path
            ),
        )
        # encoder engine 的输入 profile 是 [1,3,1024,1024]（trtexec 读 binding 确认），
        # 必须保持 nanosam 默认的 1024，传 512 会触发 TRT 输入绑定越界
        self.sam = Predictor(
            image_encoder_engine="{}/resnet18_image_encoder.engine".format(data_path),
            mask_decoder_engine="{}/mobile_sam_mask_decoder.engine".format(data_path),
        )
        self._text_cache: Dict[Tuple[str, ...], OwlEncodeTextOutput] = {}

    def encode_text_cached(
        self, prompts: List[str]
    ):
        key = tuple(prompts)
        if key not in self._text_cache:
            self._text_cache[key] = self.owl.encode_text(prompts)
        return self._text_cache[key]

    def warmup(self) -> None:
        """对 dummy 图跑一轮，把 TRT 反序列化 / cuDNN autotune 的一次性开销付掉。"""
        dummy = Image.new("RGB", (948, 768), (0, 0, 0))
        prompts = ["warmup"]
        text_encodings = self.encode_text_cached(prompts)
        self.owl.predict(
            dummy, prompts, text_encodings=text_encodings, threshold=0.1
        )
        self.sam.set_image(dummy)
        self.sam.predict(
            np.array([[100, 100], [200, 200]]), np.array([2, 3])
        )
        torch.cuda.empty_cache()
        print("[engine] warmup done", flush=True)

    def infer(
        self,
        image_pil: Image.Image,
        prompts: List[str],
        confidence_threshold: float,
    ) -> Dict:
        t0 = time.perf_counter()

        # 1. 等比缩放到高 image_height，记原图尺寸用于坐标/掩码还原
        orig_w, orig_h = image_pil.size
        resized = image_pil.resize(
            (
                round(self.image_height * orig_w / orig_h),
                self.image_height,
            )
        )
        scale_x = orig_w / resized.width
        scale_y = orig_h / resized.height

        # 2. OWL 检测（文本编码走缓存）
        t1 = time.perf_counter()
        text_encodings = self.encode_text_cached(prompts)
        output = self.owl.predict(
            resized,
            prompts,
            text_encodings=text_encodings,
            threshold=confidence_threshold,
        )
        labels = output.labels.detach().cpu().numpy()
        scores = output.scores.detach().cpu().numpy()
        boxes = output.boxes.detach().cpu().numpy()
        t2 = time.perf_counter()

        # 3. NanoSAM：全图编码一次，再对每个框解码
        self.sam.set_image(resized)
        t3 = time.perf_counter()

        results: List[Dict] = []
        for i, prompt in enumerate(prompts):
            idxs = np.where(labels == i)[0]
            if len(idxs) == 0:
                results.append(
                    {"prompt": prompt, "scores": [], "boxes": [], "masks_png_b64": []}
                )
                continue
            # 该类框按分数排序取 top-K，并过滤过小的框（resize 图坐标下面积 < 100 px²）
            idxs = idxs[np.argsort(-scores[idxs])][: self.max_detections_per_prompt]
            scores_i: List[float] = []
            boxes_i: List[List[float]] = []
            masks_i: List[str] = []
            for idx in idxs:
                x0, y0, x1, y1 = (float(v) for v in boxes[idx])
                if (x1 - x0) * (y1 - y0) < self.min_box_area:
                    continue
                hi_res_mask, _, _ = self.sam.predict(
                    np.array([[x0, y0], [x1, y1]]), np.array([2, 3])
                )
                mask = (hi_res_mask[0, 0] > 0.5).detach().cpu().numpy()
                if os.environ.get("MASK_CODEC", "rle") == "rle":
                    # 1024 分辨率直接游程编码（~5ms），桥端兜底 cv2 NEAREST 放大到原图
                    masks_i.append(_rle_encode((mask * 255).astype(np.uint8)))
                else:
                    # PNG 路径：PIL NEAREST 放大到原图（旧行为）
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                        (orig_w, orig_h), Image.NEAREST
                    )
                    masks_i.append(encode_mask(np.asarray(mask_img, dtype=np.uint8)))
                scores_i.append(float(scores[idx]))
                boxes_i.append(
                    [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                )
            results.append(
                {
                    "prompt": prompt,
                    "scores": scores_i,
                    "boxes": boxes_i,
                    "masks_png_b64": masks_i,
                }
            )
        t4 = time.perf_counter()

        return {
            "engine_ms": (t4 - t0) * 1000.0,
            "detect_ms": (t2 - t1) * 1000.0,
            "sam_encode_ms": (t3 - t2) * 1000.0,
            "sam_decode_ms": (t4 - t3) * 1000.0,
            "results": results,
        }


class YoloeModel:
    """YOLOE 后端：检测 + 实例分割一体，**不需要 NanoSAM**。

    为什么能省掉 NanoSAM：YOLOE 的掩码是从 mask 原型（32 个 160x160 的基）乘上
    每个框的系数算出来的，一次前向就全出来了，**不随框数线性增长**。而 NanoSAM 是
    逐框解码——本机实测同一张图 5 个框时，NanoSAM 那段就吃掉 84ms / 141ms（60%）。

    对外的 infer() 返回结构跟 NanoPerceptionModel **逐字段一致**，所以换后端不需要
    动 shim / realworld_detector / rgb_overlay 任何一行。字段对齐情况（本机实测）：
      · boxes  —— YOLOE 的 boxes.xyxy 本来就是原图像素坐标，跟原后端一致，不用换算
      · masks  —— YOLOE 的 masks.data 是 uint8 0/1、**直接是原图尺寸**，比原后端的
                  1024x1024 还省一次放大；直接喂 _rle_encode 即可
      · scores —— 标准 sigmoid 置信度（0~1），不像 NanoOWL 挤在 0.02~0.20。
                  **转接头那把 ScoreRemapper 尺子对这个后端是多余的**，切过来之后
                  要把 config/score_map.json 一并撤掉，否则会被二次拉伸

    词表用 set_classes 烘进模型，**PyTorch 路径下可以运行时换**（导出成 TensorRT
    engine 才会焊死）。这里跟原后端的 encode_text_cached 一样做了缓存，同一组词
    只算一次文本编码。
    """

    def __init__(
        self,
        weights: str,
        device: str,
        max_detections_per_prompt: int,
        min_box_area: float = 100.0,
    ) -> None:
        # 规则：YOLOE 引擎与 Odin 相机驱动同机运行时必须限制 OMP/BLAS 线程数，否则
        # YOLOE（纯 PyTorch，CPU 卷积由 OpenMP 占满所有核）会和 Odin 驱动的回调线程
        # 抢核，抢不到就丢帧、掉线；下面几行就是干这个的（限 4 线程即可，GPU 才是
        # YOLOE 主要算力；需在 import torch 前设置才生效，启动脚本里也设了一份兜底）。
        _n_threads = int(os.environ.get("YOLOE_TORCH_THREADS", "4"))
        os.environ.setdefault("OMP_NUM_THREADS", str(_n_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(_n_threads))
        torch.set_num_threads(_n_threads)
        print("[engine] torch 线程数限制为 %d（防止抢占 Odin 驱动的回调线程）"
              % _n_threads, flush=True)

        from ultralytics import YOLOE          # 延迟 import：nanoowl 后端不需要 ultralytics

        self.device = device
        self.max_detections_per_prompt = max_detections_per_prompt
        self.min_box_area = min_box_area
        self.model = YOLOE(weights)
        self._classes_key: Optional[Tuple[str, ...]] = None

    def set_classes_cached(self, prompts: List[str]) -> None:
        """同一组词只跑一次文本编码，跟原后端的 encode_text_cached 同一个思路。"""
        key = tuple(prompts)
        if key != self._classes_key:
            self.model.set_classes(list(prompts), self.model.get_text_pe(list(prompts)))
            self._classes_key = key

    def warmup(self) -> None:
        dummy = Image.new("RGB", (640, 480))
        self.infer(dummy, ["object"], 0.1)
        print("[engine] warmup done (yoloe)", flush=True)

    def infer(
        self,
        image_pil: Image.Image,
        prompts: List[str],
        confidence_threshold: float,
    ) -> Dict:
        t0 = time.perf_counter()
        self.set_classes_cached(prompts)
        t1 = time.perf_counter()

        # 传 PIL 而不是 numpy：ultralytics 对 numpy 数组按 BGR 解释、对 PIL 按 RGB，
        # 传 PIL 就不用操心通道顺序（搞反了分数会明显下降）。
        out = self.model.predict(
            image_pil, conf=confidence_threshold, half=True,
            device=0 if self.device == "cuda" else self.device, verbose=False,
        )[0]
        t2 = time.perf_counter()

        n = 0 if out.boxes is None else len(out.boxes)
        cls = out.boxes.cls.tolist() if n else []
        conf = out.boxes.conf.tolist() if n else []
        xyxy = out.boxes.xyxy.tolist() if n else []
        masks = out.masks.data.cpu().numpy() if (n and out.masks is not None) else None

        results: List[Dict] = []
        for i, prompt in enumerate(prompts):
            idxs = [k for k in range(n) if int(cls[k]) == i]
            # 同一类按分数排序取 top-K，口径跟原后端一致
            idxs.sort(key=lambda k: -conf[k])
            idxs = idxs[: self.max_detections_per_prompt]

            scores_i: List[float] = []
            boxes_i: List[List[float]] = []
            masks_i: List[str] = []
            for k in idxs:
                x0, y0, x1, y1 = (float(v) for v in xyxy[k])
                # 注意：原后端的 min_box_area 是在 resize 图坐标下算的，这里是原图
                # 坐标，同一个数值含义略有差别。原图更大，所以这里等效更宽松。
                if (x1 - x0) * (y1 - y0) < self.min_box_area:
                    continue
                if masks is None:
                    continue
                m = (masks[k] > 0.5).astype(np.uint8) * 255
                if os.environ.get("MASK_CODEC", "rle") == "rle":
                    masks_i.append(_rle_encode(m))
                else:
                    masks_i.append(encode_mask(m))
                scores_i.append(float(conf[k]))
                boxes_i.append([x0, y0, x1, y1])
            results.append(
                {
                    "prompt": prompt,
                    "scores": scores_i,
                    "boxes": boxes_i,
                    "masks_png_b64": masks_i,
                }
            )
        t3 = time.perf_counter()

        # 字段名保持跟原后端一致（下游 bridge/detector 在读）。YOLOE 是一体的，
        # 没有独立的 SAM 阶段：sam_encode_ms 记文本编码（同样是"每组词一次"的开销），
        # sam_decode_ms 记掩码编码 + 组装。
        return {
            "engine_ms": (t3 - t0) * 1000.0,
            "detect_ms": (t2 - t1) * 1000.0,
            "sam_encode_ms": (t1 - t0) * 1000.0,
            "sam_decode_ms": (t3 - t2) * 1000.0,
            "results": results,
        }


class EngineHTTPServer(ThreadingHTTPServer):
    """带模型引用与忙标志的 HTTP 服务器。busy 锁 acquire 失败即忙。"""

    def __init__(
        self, model, port: int, default_threshold: float, backend: str = "nanoowl"
    ) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.model = model
        self.busy = threading.Lock()
        self.default_threshold = default_threshold
        # /health 会把它报出去，好让上游（start_snownav_go2.sh）知道该不该传分数尺子：
        # nanoowl 的原始分挤在 0.02~0.20 必须拉伸，yoloe 出的已经是标准 0~1，
        # 再过一遍 ScoreRemapper 就是二次拉伸、分数全乱。
        self.backend = backend


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # 安静模式，日志走 stdout
        pass

    def _reply(self, code: int, payload: Dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._reply(200, {
                "status": "ready",
                "backend": getattr(self.server, "backend", "nanoowl"),
            })
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._reply(404, {"error": "not found"})
            return
        server = self.server
        if not server.busy.acquire(blocking=False):
            self._reply(503, {"error": "busy"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
            raw = base64.b64decode(payload["image_b64"])
            image_pil = Image.open(io.BytesIO(raw)).convert("RGB")
            prompts = payload.get("prompts")
            confidence_threshold = float(
                payload.get("confidence_threshold", server.default_threshold)
            )
            t0 = time.perf_counter()
            result = server.model.infer(image_pil, prompts, confidence_threshold)
            result["server_ms"] = (time.perf_counter() - t0) * 1000.0
            self._reply(200, result)
        except Exception as exc:  # 单次推理失败不拖垮服务
            print("[engine] infer error: {!r}".format(exc), flush=True)
            self._reply(500, {"error": str(exc)})
        finally:
            server.busy.release()


def main() -> None:
    args = parse_args()
    data_path = args.data_path or "{}/data".format(args.repo_path)
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print("[engine] loading models on {} (backend={}) ...".format(
        args.device, args.backend), flush=True)
    if args.backend == "yoloe":
        model = YoloeModel(
            args.yoloe_weights,
            args.device,
            args.max_detections_per_prompt,
            args.min_box_area,
        )
    else:
        model = NanoPerceptionModel(
            args.repo_path,
            data_path,
            args.device,
            args.image_height,
            args.max_detections_per_prompt,
            args.min_box_area,
        )
    model.warmup()
    server = EngineHTTPServer(model, args.port, args.confidence_threshold, args.backend)

    def _shutdown(signum, frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    print(
        "[engine] ready on 127.0.0.1:{}, backend={}, prompts={}".format(
            args.port, args.backend, prompts),
        flush=True,
    )
    server.serve_forever()
    print("[engine] shutdown", flush=True)


if __name__ == "__main__":
    main()
