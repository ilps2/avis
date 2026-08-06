#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""剪辑规划效果评估：规则 -> 剪辑清单 -> 与真值窗口对比。

指标：
- Clip Precision@k：规划出的片段里有多少与真值重叠 >= 1 秒
- GT Recall：真值窗口被至少一条规划片段覆盖的比例
- 边界误差：规划片段起点/终点离最近真值边界的平均秒数
- 语音覆盖：规划片段的平均语音秒数（编辑可用性）

真值 = 代理真值（对象层窗口 + large-v3 ASR 关键词窗口），
与规划器信号同源，属于"自证"测试；正式评估需人工逐帧标注。
"""

import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_layer import load_index, plan_clips

IDX = Path(__file__).resolve().parent / "demo_clip03_v3"
RULES = [
    {"name": "口红+讲解", "text": "口红", "objects": ["lipstick"],
     "keywords": ["口红"]},
    {"name": "价格+优惠", "text": "价格 优惠 下单", "objects": ["price tag", "price"],
     "keywords": ["下单", "礼赠", "价格", "四百"]},
    {"name": "二号链接", "text": "二号链接", "objects": [],
     "keywords": ["二号"]},
    {"name": "粉底气垫", "text": "粉底 气垫 底妆", "objects": [],
     "keywords": ["粉", "气垫", "底妆"]},
]


def load(idx):
    _, objects, scenes, asr, emb = load_index(idx)
    return objects, scenes, asr, emb


def build_gt(objects, scenes, asr):
    def obj_windows(labels):
        return [(o["t0"], o["t1"]) for o in objects if o["label"] in labels]

    def asr_windows(kws):
        return [(a["start"], a["end"]) for a in asr if any(k in a["text"] for k in kws)]

    speech_win = []
    for a in asr:
        speech_win.append((a["start"], a["end"]))
    lip_visual = obj_windows({"lipstick"})
    return {
        "口红+讲解": lip_visual + asr_windows(["口红"]),
        "价格+优惠": obj_windows({"price tag", "price"})
        + asr_windows(["下单", "礼赠", "价格", "四百"]),
        "二号链接": asr_windows(["二号"]),
        "粉底气垫": asr_windows(["粉气垫", "底妆"]),
    }


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def nearest_boundary(t, bounds):
    return min(bounds, key=lambda b: abs(b - t))


def main():
    objects, scenes, asr, emb = load(IDX)
    gt_all = build_gt(objects, scenes, asr)
    print(f"{'规则':<10} {'clips':>5} {'P@k':>6} {'GT召回':>6} {'边界误差(s)':>10} "
          f"{'均长(s)':>8} {'语音(s)':>7}")
    rows = {}
    for rule in RULES:
        with contextlib.redirect_stdout(sys.stderr):
            plan = plan_clips(
                IDX, text=rule["text"], object_labels=rule["objects"],
                require_speech=True, min_dur=3.0, max_clips=10,
                target_dur=20.0, keywords=rule.get("keywords"))
        clips = plan["clips"]
        gt = gt_all[rule["name"]]
        if not clips:
            print(f"{rule['name']:<10} {0:>5} {'-':>6} {'-':>6} {'-':>10}")
            rows[rule["name"]] = {"p": 0, "r": 0, "be": None, "dur": 0, "sp": 0}
            continue
        hits = [c for c in clips if any(overlap((c["t0"], c["t1"]), g) >= 1.0 for g in gt)]
        p = len(hits) / len(clips)
        covered = sum(1 for g in gt
                      if any(overlap((c["t0"], c["t1"]), g) >= 1.0 for c in clips))
        r = covered / len(gt) if gt else 0.0
        bounds = [b for g in gt for b in g]
        be = sum(abs(c["t0"] - nearest_boundary(c["t0"], bounds))
                 + abs(c["t1"] - nearest_boundary(c["t1"], bounds))
                 for c in clips) / (2 * len(clips)) if bounds else 0.0
        dur = sum(c["duration"] for c in clips) / len(clips)
        sp = sum(c["speech_sec"] for c in clips) / len(clips)
        rows[rule["name"]] = {"p": p, "r": r, "be": be, "dur": dur, "sp": sp}
        print(f"{rule['name']:<10} {len(clips):>5} {p:>6.2f} {r:>6.2f} "
              f"{be:>10.2f} {dur:>8.1f} {sp:>7.1f}")
    avg = {k: sum(v[k] for v in rows.values() if v[k] is not None) / len(rows)
           for k in ["p", "r", "be"]}
    print(f"\n平均: P@k={avg['p']:.2f} | GT召回={avg['r']:.2f} | 边界误差={avg['be']:.2f}s")
    out = Path(__file__).resolve().parent / "eval_clips_results.json"
    out.write_text(json.dumps({"rows": rows, "avg": avg, "gt": {
        k: [list(x) for x in v] for k, v in gt_all.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out)


if __name__ == "__main__":
    main()
