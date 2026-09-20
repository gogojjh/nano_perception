#!/usr/bin/env python3
"""掩码一致性校验（V3）：验证加速改动不改变掩码数值。

模块 A（编解码级）：对模拟真实掩码（1024x1024 bool 放大到 1600x1296），
  三种编码路径（pil / fast=torchvision / rle）编码→解码后逐像素一致 + 编码耗时。
模块 B（模型级）：SAM_PREPROC=legacy vs fast 在同进程切换 env 跑完整 infer，
  对比每框掩码 IoU 与像素一致率（浮点差异允许，IoU>0.95 判通过）。

运行环境：conda nano_perception（LD_PRELOAD/PYTHONPATH 同 start_nano_perception.sh）
用法：
  python check_mask_consistency.py --images raw1.jpg raw2.jpg --out-dir results/mask_check
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="掩码一致性校验")
    parser.add_argument("--images", nargs="+", default=None,
                        help="模块 B 用实机原帧 jpg（默认无，只跑模块 A）")
    parser.add_argument("--prompts", default=None,
                        help="逗号分隔类别，默认读 objectnav-realworld-prompts.txt")
    parser.add_argument("--confidence-threshold", type=float, default=0.08)
    parser.add_argument("--repo-path", default="/home/cmit/robohike_ws/src/nano_perception")
    parser.add_argument("--out-dir", default=None,
                        help="默认 results/mask_check/")
    return parser.parse_args()


def rle_decode(b64: str) -> np.ndarray:
    """与 nano_perception_bridge.decode_mask 的 rle 分支同逻辑（纯 numpy 版）。

    b64: 'rle:' 前缀 + base64(uint32 小端游程对)
    返回原尺寸 uint8 0/255 掩码（后续 >127 判 bool 与 bridge 一致）。
    """
    import base64
    raw = base64.b64decode(b64[len("rle:"):])
    runs = np.frombuffer(raw, dtype="<u4")
    h = int(runs[0])
    w = int(runs[1])
    total = int(np.sum(runs[2:]))
    if total != h * w:
        raise ValueError("RLE 解出 {} 像素，期望 {}".format(total, h * w))
    flat = np.empty(h * w, dtype=np.uint8)
    pos = 0
    val = 0  # 段值从 0 开始交替（段 0 必为 0 值段，见 engine 编码）
    for i in range(2, len(runs)):
        n = int(runs[i])
        flat[pos:pos + n] = val
        pos += n
        val = 255 if val == 0 else 0
    return flat.reshape(h, w)


def png_decode_pil(b64: str) -> np.ndarray:
    """PIL 解码 PNG（与 bridge cv2.imdecode 灰度 >127 等价，PNG 无损）。"""
    import base64
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return np.asarray(img.convert("L"), dtype=np.uint8)


def make_fake_masks(orig_size, n=8):
    """模拟真实分割掩码：SAM 输出 1024x1024 bool → PIL NEAREST 放大到原图。

    返回 (1024 分辨率掩码列表, 原图分辨率期望列表)。形状覆盖：
    大实心 blob、小 blob、细长条、空心环（逐像素一致最严苛样本）。
    """
    oh, ow = orig_size
    lowres, expects = [], []
    rng = np.random.RandomState(42)
    for k in range(n):
        m = np.zeros((1024, 1024), dtype=np.bool_)
        if k % 4 == 0:  # 大椭圆 blob
            yy, xx = np.mgrid[0:1024, 0:1024]
            cy, cx = 100 + rng.randint(800), 100 + rng.randint(800)
            m = ((yy - cy) / (80 + rng.randint(300))) ** 2 + \
                ((xx - cx) / (60 + rng.randint(200))) ** 2 < 1
        elif k % 4 == 1:  # 实心矩形
            y0, x0 = rng.randint(0, 700, 2)
            m[y0:y0 + 200 + rng.randint(300), x0:x0 + 150 + rng.randint(300)] = True
        elif k % 4 == 2:  # 细长条（NEAREST 放大易断裂）
            y0, x0 = rng.randint(0, 900, 2)
            m[y0:y0 + 8, x0:x0 + 600] = True
        else:  # 空心环
            yy, xx = np.mgrid[0:1024, 0:1024]
            cy, cx = rng.randint(200, 824), rng.randint(200, 824)
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            m = (r2 < 250 ** 2) & (r2 > 120 ** 2)
        lowres.append(m)
        # 期望：PIL NEAREST 放大（与桥端 cv2 NEAREST 数学一致）
        expects.append(
            np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
                (ow, oh), Image.NEAREST)) > 127
        )
    return lowres, expects


def module_a(engine_mod, out_dir):
    """编解码级：pil/fast/rle 三路径逐像素一致 + 编码耗时表。

    rle 路径按新语义：1024 分辨率编码，解码后模拟桥端兜底 NEAREST 放大再对比。
    """
    print("\n==== 模块 A：掩码编解码一致性（1024 -> 1600x1296）====")
    lowres_masks, expects = make_fake_masks((1296, 1600))
    rows = []
    for codec in ("pil", "fast", "rle"):
        times = []
        for low, exp in zip(lowres_masks, expects):
            t0 = time.perf_counter()
            if codec == "rle":
                b64 = engine_mod._rle_encode((low * 255).astype(np.uint8))
                back = rle_decode(b64) > 127
                # 模拟桥端兜底：cv2 NEAREST 放大（与 PIL NEAREST 数学一致）
                back = np.asarray(Image.fromarray((back * 255).astype(np.uint8)).resize(
                    (exp.shape[1], exp.shape[0]), Image.NEAREST)) > 127
            else:
                os.environ["MASK_CODEC"] = codec
                b64 = engine_mod.encode_mask((exp * 255).astype(np.uint8))
                back = png_decode_pil(b64) > 127
            times.append((time.perf_counter() - t0) * 1000.0)
            if back.shape != exp.shape or not np.array_equal(back, exp):
                print("  [FAIL] codec={} 掩码不一致".format(codec))
                return False
        rows.append({
            "codec": codec,
            "encode_ms_mean": round(float(np.mean(times)), 3),
            "encode_ms_max": round(max(times), 3),
            "pixel_exact": True,
        })
        print("  codec={}: 编码 mean {:.3f} ms / max {:.3f} ms，逐像素一致 ✓".format(
            codec, np.mean(times), max(times)))
    with open(os.path.join(out_dir, "module_a.json"), "w") as f:
        json.dump({"masks": len(lowres_masks), "results": rows}, f, indent=2)
    return True


def module_b(engine_mod, model, images, prompts, threshold, out_dir):
    """模型级：SAM_PREPROC legacy vs fast 掩码 IoU 与像素一致率。"""
    print("\n==== 模块 B：SAM_PREPROC legacy vs fast（模型级）====")
    pairs = []
    for img_path in images:
        image_pil = Image.open(img_path).convert("RGB")
        masks_by_mode = {}
        for mode in ("legacy", "fast"):
            os.environ["SAM_PREPROC"] = mode
            result = model.infer(image_pil, prompts, threshold)
            ms = []
            for r in result["results"]:
                for mb64 in r["masks_png_b64"]:
                    ms.append(rle_decode(mb64) > 127)
            masks_by_mode[mode] = ms
        if len(masks_by_mode["legacy"]) != len(masks_by_mode["fast"]):
            print("  [FAIL] {} 两模式检测框数不同: {} vs {}".format(
                os.path.basename(img_path),
                len(masks_by_mode["legacy"]), len(masks_by_mode["fast"])))
            return False
        ious = []
        for m1, m2 in zip(masks_by_mode["legacy"], masks_by_mode["fast"]):
            inter = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            ious.append(float(inter) / max(union, 1))
            pairs.append({
                "image": os.path.basename(img_path),
                "iou": round(float(ious[-1]), 4),
                "pixel_agree": round(float((m1 == m2).mean()), 4),
            })
        print("  {}: {} 框，IoU = {}".format(
            os.path.basename(img_path), len(ious),
            ", ".join("{:.4f}".format(i) for i in ious)))
    min_iou = min(p["iou"] for p in pairs)
    mean_iou = float(np.mean([p["iou"] for p in pairs]))
    ok = min_iou > 0.95
    print("  IoU mean {:.4f} / min {:.4f} -> {}".format(
        mean_iou, min_iou, "通过（>0.95）✓" if ok else "不通过 ✗"))
    with open(os.path.join(out_dir, "module_b.json"), "w") as f:
        json.dump({"pairs": pairs, "min_iou": min_iou, "mean_iou": mean_iou,
                   "passed": ok}, f, indent=2)
    return ok


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.repo_path)
    from nano_perception_engine import NanoPerceptionModel

    out_dir = args.out_dir or os.path.join(args.repo_path, "results", "mask_check")
    os.makedirs(out_dir, exist_ok=True)

    import nano_perception_engine as engine_mod

    ok_a = module_a(engine_mod, out_dir)

    ok_b = True
    if args.images:
        prompts = None
        if args.prompts:
            prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
        else:
            with open(os.path.join(args.repo_path, "objectnav-realworld-prompts.txt")) as f:
                prompts = [p.strip() for p in f.read().replace("\n", ",").split(",") if p.strip()]
        data_path = os.path.join(args.repo_path, "data")
        model = NanoPerceptionModel(args.repo_path, data_path, "cuda", 768, 6)
        model.warmup()
        os.environ["MASK_CODEC"] = "rle"  # 对比掩码数值，用无损编码
        ok_b = module_b(engine_mod, model, args.images, prompts,
                        args.confidence_threshold, out_dir)

    print("\n==== 总判定：模块 A {}，模块 B {} ====".format(
        "通过 ✓" if ok_a else "失败 ✗", "通过 ✓" if ok_b else "失败 ✗"))
    sys.exit(0 if (ok_a and ok_b) else 1)


if __name__ == "__main__":
    main()
