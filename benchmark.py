#!/usr/bin/env python3
"""NanoOWL+NanoSAM 分模块延迟基准（conda nano_perception env 下运行）。

直接调用引擎的 infer()，不走 HTTP，测纯推理耗时。
输出 mean / median / p95 的 engine_ms / detect_ms / sam_encode_ms / sam_decode_ms。

用法：
  python benchmark.py --image test.jpg --prompts "person, chair" --iters 20
"""
import argparse
import json
import statistics
import sys
import time
from typing import Dict, List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NanoOWL+NanoSAM 延迟基准")
    parser.add_argument("--image", required=True, help="测试图片路径")
    parser.add_argument(
        "--prompts",
        default="person, chair, table, bottle, garbage can, door, box, monitor, fire hydrant",
    )
    parser.add_argument("--iters", type=int, default=20, help="正式计时轮数")
    parser.add_argument("--warmup", type=int, default=2, help="预热轮数")
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--repo-path", default="/home/cmit/robohike_ws/src/nano_perception")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="汇总 JSON 输出路径（可选）")
    return parser.parse_args()


def summarize_dict(values: List[float]) -> Dict[str, float]:
    mean = sum(values) / len(values)
    median = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95) - 1] if len(values) >= 20 else max(values)
    return {"mean_ms": round(mean, 2), "median_ms": round(median, 2), "p95_ms": round(p95, 2)}


def summarize(name: str, values: List[float]) -> str:
    d = summarize_dict(values)
    return "{}: mean {:.2f} ms | median {:.2f} ms | p95 {:.2f} ms".format(
        name, d["mean_ms"], d["median_ms"], d["p95_ms"]
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.repo_path)
    from nano_perception_engine import NanoPerceptionModel

    data_path = args.data_path or "{}/data".format(args.repo_path)
    print("[bench] 加载模型 ...")
    t0 = time.perf_counter()
    model = NanoPerceptionModel(args.repo_path, data_path, args.device, 768, 6)
    model.warmup()
    print("[bench] 模型加载+预热 {:.1f} s".format(time.perf_counter() - t0))

    image_pil = Image.open(args.image).convert("RGB")
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print("[bench] 图 {}x{}, prompts={}".format(*image_pil.size, prompts))

    for i in range(args.warmup):
        model.infer(image_pil, prompts, args.confidence_threshold)
    print("[bench] 预热 {} 轮完成，开始正式计时 ...".format(args.warmup))

    metrics: Dict[str, List[float]] = {
        "engine_ms": [],
        "detect_ms": [],
        "sam_encode_ms": [],
        "sam_decode_ms": [],
    }
    for i in range(args.iters):
        result = model.infer(image_pil, prompts, args.confidence_threshold)
        for key in metrics:
            metrics[key].append(result[key])
        n_det = sum(len(r["boxes"]) for r in result["results"])
        print(
            "  [{:02d}] engine {:.1f} ms (detect {:.1f} / enc {:.1f} / dec {:.1f}), "
            "{} 个检测".format(
                i,
                result["engine_ms"],
                result["detect_ms"],
                result["sam_encode_ms"],
                result["sam_decode_ms"],
                n_det,
            )
        )

    print("---- 汇总（{} 轮）----".format(args.iters))
    for key in metrics:
        print(summarize(key, metrics[key]))

    if args.output:
        out = {
            "image": args.image,
            "prompts": prompts,
            "iters": args.iters,
            "width": image_pil.width,
            "height": image_pil.height,
            "confidence_threshold": args.confidence_threshold,
            "summary": {key: summarize_dict(metrics[key]) for key in metrics},
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("[bench] 汇总已写入", args.output)


if __name__ == "__main__":
    main()
