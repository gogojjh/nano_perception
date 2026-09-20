#!/usr/bin/env python3
"""多档置信度阈值扫描：在原始相机帧上评估不同 confidence_threshold 的召回/误检。

fast 模式（默认）：每帧只做一次 OWL 图像编码（TRT），各档阈值只重跑 decode
（CPU 毫秒级），跳过 SAM——数秒出「阈值-空帧率-每类框数-分数分布」表。
--full 模式：逐档阈值走完整 model.infer()（含 SAM），在少量帧上验证低阈值时
端到端延迟的上浮（框数激增 → SAM 解码变多）。

过滤顺序与 nano_perception_engine.infer 一致：先每类按分数降序取 top-K，
再按 min_box_area 过滤小框。raw_count = top-K 截断前该类框数，final_count = 过滤后。

运行环境：conda nano_perception（LD_PRELOAD/PYTHONPATH 同 start_nano_perception.sh）
用法：
  python threshold_scan.py --images raw1.jpg raw2.jpg --thresholds "0.02,0.05,0.08,0.1"
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

# 与 nano_perception_bridge.PALETTE_BGR 同色系，但 PIL 用 RGB 顺序
PALETTE_RGB = [
    (220, 20, 60),  # person       红
    (75, 180, 60),  # chair        绿
    (48, 130, 245),  # table        橙
    (25, 225, 255),  # bottle       黄
    (60, 20, 200),  # garbage can  紫
    (255, 105, 180),  # door         粉
    (250, 120, 25),  # box          蓝
    (255, 255, 0),  # monitor      青
    (100, 100, 100),  # fire hydrant 灰
]

DEFAULT_PROMPTS_FILE = "objectnav-realworld-prompts.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多档置信度阈值扫描")
    parser.add_argument("--images", nargs="+", required=True, help="原始相机帧 jpg 列表")
    parser.add_argument(
        "--prompts",
        default=None,
        help="逗号分隔类别，默认读 objectnav-realworld-prompts.txt",
    )
    parser.add_argument("--thresholds", default="0.02,0.05,0.08,0.1")
    parser.add_argument("--min-box-area", type=float, default=100.0,
                        help="小框面积过滤（resize 后图像坐标 px²）")
    parser.add_argument("--max-detections-per-prompt", type=int, default=6)
    parser.add_argument("--image-height", type=int, default=768, help="输入等比缩放后的高")
    parser.add_argument("--repo-path", default="/home/cmit/robohike_ws/src/nano_perception")
    parser.add_argument("--data-path", default=None, help="engine 目录，默认 <repo>/data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None, help="默认 results/threshold_scan/")
    parser.add_argument("--full", action="store_true",
                        help="逐档走完整 infer()（含 SAM），测端到端延迟")
    return parser.parse_args()


def load_prompts(arg_prompts: str, repo_path: str) -> List[str]:
    if arg_prompts:
        return [p.strip() for p in arg_prompts.split(",") if p.strip()]
    with open(os.path.join(repo_path, DEFAULT_PROMPTS_FILE), encoding="utf-8") as f:
        raw = f.read()
    return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]


def rescale_boxes(boxes: np.ndarray, scale_x: float, scale_y: float) -> List[List[float]]:
    return [
        [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
        for x0, y0, x1, y1 in boxes
    ]


def filter_like_engine(
    labels: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    prompts: List[str],
    max_detections_per_prompt: int,
    min_box_area: float,
) -> Tuple[int, List[Dict]]:
    """复刻 engine infer 的过滤顺序，返回 (raw_count, 最终检测列表)。"""
    raw_count = 0
    detections: List[Dict] = []
    for i, prompt in enumerate(prompts):
        idxs = np.where(labels == i)[0]
        raw_count += len(idxs)
        if len(idxs) == 0:
            continue
        idxs = idxs[np.argsort(-scores[idxs])][:max_detections_per_prompt]
        for idx in idxs:
            x0, y0, x1, y1 = (float(v) for v in boxes[idx])
            if (x1 - x0) * (y1 - y0) < min_box_area:
                continue
            detections.append(
                {
                    "label": prompt,
                    "score": float(scores[idx]),
                    "box_resized": [x0, y0, x1, y1],
                }
            )
    return raw_count, detections


def draw_boxes(
    image: Image.Image, detections: List[Dict]
) -> Image.Image:
    """把检测框画回原图，框上是 'label score'。"""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for d in detections:
        color = PALETTE_RGB[d["label_index"] % len(PALETTE_RGB)]
        x0, y0, x1, y1 = d["box_orig"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        draw.text(
            (x0 + 2, max(y0 - 16, 2)),
            "{} {:.2f}".format(d["label"], d["score"]),
            fill=color,
        )
    return vis


def summarize_threshold(
    detections: List[List[Dict]], threshold: float
) -> Dict:
    """一档阈值的汇总：总框数、空帧率、每类框数、分数分布。"""
    all_scores = [d["score"] for frame in detections for d in frame]
    per_label: Dict[str, int] = {}
    for frame in detections:
        for d in frame:
            per_label[d["label"]] = per_label.get(d["label"], 0) + 1
    buckets = {"[0,0.05)": 0, "[0.05,0.1)": 0, "[0.1,0.2)": 0, ">=0.2": 0}
    for s in all_scores:
        if s < 0.05:
            buckets["[0,0.05)"] += 1
        elif s < 0.1:
            buckets["[0.05,0.1)"] += 1
        elif s < 0.2:
            buckets["[0.1,0.2)"] += 1
        else:
            buckets[">=0.2"] += 1
    return {
        "threshold": threshold,
        "empty_frames": sum(1 for frame in detections if not frame),
        "total_frames": len(detections),
        "total_boxes": len(all_scores),
        "per_label": per_label,
        "score_min": round(min(all_scores), 4) if all_scores else None,
        "score_median": round(float(np.median(all_scores)), 4) if all_scores else None,
        "score_max": round(max(all_scores), 4) if all_scores else None,
        "score_buckets": buckets,
    }


def print_table(summaries: List[Dict], prompts: List[str]) -> None:
    print("\n==== 各档阈值汇总 ====")
    header = "{:>8} {:>8} {:>10} {:>10} {:>12} {:>12}".format(
        "阈值", "总框", "空帧", "min", "median", "max"
    )
    print(header)
    for s in summaries:
        empty = "{}/{}".format(s["empty_frames"], s["total_frames"])
        print(
            "{:>8.3f} {:>8} {:>10} {:>10} {:>12} {:>12}".format(
                s["threshold"],
                s["total_boxes"],
                empty,
                s["score_min"],
                s["score_median"],
                s["score_max"],
            )
        )
    print("\n每类框数（列 = 阈值档）:")
    thresholds_str = "".join("{:>8.3f}".format(s["threshold"]) for s in summaries)
    print("{:>14}{}".format("类别", thresholds_str))
    for i, p in enumerate(prompts):
        row = "".join(
            "{:>8}".format(s["per_label"].get(p, 0)) for s in summaries
        )
        print("{:>14}{}".format(p, row))
    print("\n分数分桶（调低阈值后多出来的框落在哪个分数段）:")
    for s in summaries:
        b = s["score_buckets"]
        print(
            "  阈值 {:.3f}: {}（其中 <0.05: {}，[0.05,0.1): {}，[0.1,0.2): {}，>=0.2: {}）".format(
                s["threshold"], s["total_boxes"],
                b["[0,0.05)"], b["[0.05,0.1)"], b["[0.1,0.2)"], b[">=0.2"],
            )
        )


def run_fast(
    model, images: List[str], prompts: List[str], thresholds: List[float],
    max_detections_per_prompt: int, min_box_area: float, image_height: int,
    out_dir: str,
) -> Tuple[List[Dict], List[Dict]]:
    """每帧一次编码，逐档重跑 decode。返回 (每档汇总, 逐帧明细)。"""
    text_enc = model.encode_text_cached(prompts)
    # per_threshold[ti] = 每帧最终检测列表
    per_threshold: List[List[List[Dict]]] = [[] for _ in thresholds]
    raw_counts: List[List[int]] = [[] for _ in thresholds]
    per_frame: List[Dict] = []

    for img_path in images:
        image_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image_pil.size
        resized = image_pil.resize(
            (round(image_height * orig_w / orig_h), image_height)
        )
        scale_x = orig_w / resized.width
        scale_y = orig_h / resized.height
        t0 = time.perf_counter()
        img_tensor = model.owl.image_preprocessor.preprocess_pil_image(resized)
        rois = torch.tensor(
            [[0, 0, resized.width, resized.height]],
            dtype=img_tensor.dtype,
            device=img_tensor.device,
        )
        enc = model.owl.encode_rois(img_tensor, rois, pad_square=True)
        enc_ms = (time.perf_counter() - t0) * 1000.0

        frame_dets_by_thresh: List[List[Dict]] = []
        for ti, thresh in enumerate(thresholds):
            out = model.owl.decode(enc, text_enc, thresh)
            labels = out.labels.detach().cpu().numpy()
            scores = out.scores.detach().cpu().numpy()
            boxes = out.boxes.detach().cpu().numpy()
            raw_count, detections = filter_like_engine(
                labels, scores, boxes, prompts,
                max_detections_per_prompt, min_box_area,
            )
            for d in detections:
                d["label_index"] = prompts.index(d["label"])
                d["box_orig"] = rescale_boxes(
                    np.array([d["box_resized"]]), scale_x, scale_y
                )[0]
            raw_counts[ti].append(raw_count)
            per_threshold[ti].append(detections)
            frame_dets_by_thresh.append(detections)

        if out_dir:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for ti, thresh in enumerate(thresholds):
                vis = draw_boxes(image_pil, frame_dets_by_thresh[ti])
                vis.save(
                    os.path.join(
                        out_dir, "{}_thr{:.3f}.jpg".format(base, thresh)
                    )
                )
        per_frame.append(
            {
                "image": os.path.basename(img_path),
                "per_threshold": [
                    {
                        "threshold": thresholds[ti],
                        "raw_count": raw_counts[ti][-1],
                        "detections": per_threshold[ti][-1],
                    }
                    for ti in range(len(thresholds))
                ],
            }
        )
        print(
            "[{}] 编码 {:.0f} ms，raw 框数 {}，final 框数 {}".format(
                os.path.basename(img_path),
                enc_ms,
                [raw_counts[ti][-1] for ti in range(len(thresholds))],
                [len(per_threshold[ti][-1]) for ti in range(len(thresholds))],
            )
        )
    summaries = [
        summarize_threshold(per_threshold[ti], thresholds[ti])
        for ti in range(len(thresholds))
    ]
    return summaries, per_frame


def run_full(
    model, images: List[str], prompts: List[str], thresholds: List[float],
) -> List[Dict]:
    """逐档走完整 infer()（含 SAM），返回每档汇总（含 engine_ms）。"""
    summaries: List[Dict] = []
    for thresh in thresholds:
        all_dets: List[List[Dict]] = []
        latencies: List[float] = []
        for img_path in images:
            image_pil = Image.open(img_path).convert("RGB")
            result = model.infer(image_pil, prompts, thresh)
            latencies.append(result["engine_ms"])
            dets = [
                {"label": r["prompt"], "score": float(s)}
                for r in result["results"]
                for s in r["scores"]
            ]
            all_dets.append(dets)
            print(
                "[full] {} 阈值 {:.3f}: engine {:.1f} ms，{} 框".format(
                    os.path.basename(img_path), thresh,
                    result["engine_ms"], len(dets),
                )
            )
        s = summarize_threshold(all_dets, thresh)
        s["engine_ms_mean"] = round(float(np.mean(latencies)), 1)
        s["engine_ms_max"] = round(max(latencies), 1)
        summaries.append(s)
    return summaries


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.repo_path)
    from nano_perception_engine import NanoPerceptionModel

    prompts = load_prompts(args.prompts, args.repo_path)
    thresholds = sorted(float(t) for t in args.thresholds.split(",") if t.strip())
    data_path = args.data_path or "{}/data".format(args.repo_path)
    out_dir = args.out_dir or os.path.join(args.repo_path, "results", "threshold_scan")
    os.makedirs(out_dir, exist_ok=True)

    print(
        "[scan] 帧数={}，prompts={}，阈值={}".format(
            len(args.images), prompts, thresholds
        )
    )
    model = NanoPerceptionModel(
        args.repo_path,
        data_path,
        args.device,
        args.image_height,
        args.max_detections_per_prompt,
        args.min_box_area,
    )
    model.warmup()

    per_frame: List[Dict] = []
    if args.full:
        summaries = run_full(model, args.images, prompts, thresholds)
    else:
        summaries, per_frame = run_fast(
            model, args.images, prompts, thresholds,
            args.max_detections_per_prompt, args.min_box_area,
            args.image_height, out_dir,
        )

    print_table(summaries, prompts)

    out = {
        "images": args.images,
        "prompts": prompts,
        "thresholds": thresholds,
        "min_box_area": args.min_box_area,
        "max_detections_per_prompt": args.max_detections_per_prompt,
        "mode": "full" if args.full else "fast",
        "summary": summaries,
        "per_frame": per_frame,
    }
    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[scan] 汇总已写入 {}，可视化在 {}".format(out_json, out_dir))


if __name__ == "__main__":
    main()
