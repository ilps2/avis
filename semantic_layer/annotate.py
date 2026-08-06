#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人工标注工作流（评估第 2 层）：

对每个视频 x 每条规则，自动生成候选剪辑片段（mp4 预览）和一个
labels.json。人工看片段后把 label 改成 1（值得剪）/ 0（不值得剪），
再用 eval_clips 换成人工真值跑 P/R。

用法:
  python3 annotate.py <idx_dir> --video <video.mp4> -o <out_dir>
      --text "口红" --objects lipstick --keywords 口红 --max-clips 10
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_layer import plan_clips


def cut_previews(video: Path, out_dir: Path, plan, padding=0.5):
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, c in enumerate(plan.get("clips", []), 1):
        t0 = max(0.0, c["t0"] - padding)
        dur = max(0.5, (c["t1"] + padding) - t0)
        fname = f"clip_{i:02d}_{t0:.0f}-{t0+dur:.0f}s.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(t0), "-i", str(video), "-t", str(dur),
            "-c", "copy", "-avoid_negative_ts", "make_zero", "-y",
            str(out_dir / fname),
        ], check=True)
        entries.append({
            "file": str(out_dir / fname), "t0": round(t0, 2),
            "t1": round(t0 + dur, 2), "label": None,
            "speech": c.get("speech_text", ""),
            "layers": c.get("layers", ""),
        })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idx")
    ap.add_argument("--video", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--objects", default=None)
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-clips", type=int, default=10)
    args = ap.parse_args()

    objects = [x.strip() for x in args.objects.split(",") if x.strip()] if args.objects else None
    keywords = [x.strip() for x in args.keywords.split(",") if x.strip()] if args.keywords else None
    # 高召回版（无关键词）+ 精确版（有关键词）合并去重，标注者看到全部候选
    plan_a = plan_clips(Path(args.idx), text=args.text, object_labels=objects,
                        require_speech=True, min_dur=args.min_dur,
                        max_clips=args.max_clips * 2)
    plan_b = plan_clips(Path(args.idx), text=args.text, object_labels=objects,
                        require_speech=True, min_dur=args.min_dur,
                        max_clips=args.max_clips * 2, keywords=keywords) if keywords else None
    seen, clips = set(), []
    for c in plan_a["clips"] + (plan_b["clips"] if plan_b else []):
        key = (round(c["t0"], 1), round(c["t1"], 1))
        if key not in seen:
            seen.add(key)
            clips.append(c)
    clips.sort(key=lambda c: c["t0"])
    plan = {"clips": clips[:args.max_clips], "stats": {"total_candidates": len(clips)}}
    out = Path(args.out)
    entries = cut_previews(Path(args.video), out, plan)
    labels = {"video": str(Path(args.video)), "text": args.text,
              "objects": objects, "keywords": keywords, "entries": entries}
    (out / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(entries)} 个候选片段已导出到 {out}")
    print("标注方法：播放每个 mp4，把 labels.json 里对应 label 改成 1/0（1=值得剪）")


if __name__ == "__main__":
    main()
