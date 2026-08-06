#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估语义层检索精度：10 查询 × 3 ASR 档位 × 3 检索通道。

真值 (ground truth) 由可靠信号构造（代理真值）：
- 对象类查询：GroundingDINO 对象层的出现窗口
- 语音类查询：large-v3 ASR 关键词窗口
- 场景类查询：BLIP 场景描述/CLIP 场景标签窗口

指标：窗口级 Precision@5 和 Recall@5（命中 = 与真值窗口重叠 >= 1 秒）。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semantic_layer import CLIPEmbedder, TextEmbedder

IDX = Path(__file__).resolve().parent / "examples" / "demo_clip03_v3"
TOP_K = 5
MERGE_GAP = 3.0
MAX_SPAN = 15.0


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def load_index(asr_key):
    objects = load_jsonl(IDX / "objects.jsonl")
    scenes = load_jsonl(IDX / "scenes.jsonl")
    emb = np.load(IDX / "embeddings.npz")
    vdir = IDX / "asr_variants"
    if (vdir / f"asr_{asr_key}.jsonl").exists():
        asr = load_jsonl(vdir / f"asr_{asr_key}.jsonl")
        asr_emb = np.load(vdir / f"asr_{asr_key}_emb.npy")
    else:
        print(f"[eval] asr_variants/{asr_key} 不存在，回退索引内置 ASR")
        asr = load_jsonl(IDX / "asr.jsonl")
        asr_emb = emb["asr_emb"]
    return objects, scenes, asr, emb, asr_emb


def build_gt(objects, scenes, asr_large):
    def obj_windows(label):
        return [(o["t0"], o["t1"]) for o in objects if o["label"] == label]

    def asr_windows(kws):
        return [(a["start"], a["end"]) for a in asr_large
                if any(k in a["text"] for k in kws)]

    def scene_windows(cond):
        return [(s["t0"], s["t1"]) for s in scenes if cond(s)]

    lip_visual = obj_windows("lipstick")
    return {
        "q1_lipstick_zh": {"text": "口红", "gt": lip_visual + asr_windows(["口红"])},
        "q2_lipstick_en": {"text": "lipstick", "gt": lip_visual + asr_windows(["口红"])},
        "q3_lipstick_img": {"image": "thumbs/track_15.jpg", "gt": lip_visual},
        "q4_price": {"text": "价格 优惠 下单", "gt": obj_windows("price tag")
                     + asr_windows(["下单", "礼赠", "四百", "价格"])},
        "q5_link2": {"text": "二号链接", "gt": asr_windows(["二号"])},
        "q6_foundation": {"text": "粉底 气垫 底妆", "gt": asr_windows(["粉气垫", "底妆"])},
        "q7_iceoolong": {"text": "冰乌龙 推荐色号", "gt": asr_windows(["冰乌龙", "推荐"])},
        "q8_gift": {"text": "送口红 下单福利", "gt": asr_windows(["口红", "礼赠", "下单"])},
        "q9_box_img": {"image": "thumbs/track_9.jpg", "gt": obj_windows("box")
                       + scene_windows(lambda s: "box" in s["caption"])},
        "q10_phone": {"text": "手机 phone", "gt": scene_windows(
            lambda s: "phone" in s["caption"] or "phone" in s["tags"])},
        "q11_lipstick_both": {"text": "口红", "image": "thumbs/track_15.jpg",
                              "gt": lip_visual + asr_windows(["口红"])},
    }


def mk_hit(t0, t1, score, layer, text):
    return {"t0": float(t0), "t1": float(t1), "score": float(score),
            "layer": layer, "text": text}


def merge_hits(hits):
    hits = sorted(hits, key=lambda h: h["score"], reverse=True)
    merged = []
    for h in hits:
        if (merged and h["t0"] <= merged[-1]["t1"] + MERGE_GAP
                and h["t1"] - merged[-1]["t0"] <= MAX_SPAN):
            m = merged[-1]
            m["t1"] = max(m["t1"], h["t1"])
            m["score"] = max(m["score"], h["score"])
            m["layer"] += "+" + h["layer"]
        else:
            merged.append(dict(h))
    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged[:TOP_K]


def search_text(te, q, objects, scenes, asr, emb, asr_emb):
    hits = []
    qe = te.embed([q])[0]
    if len(objects):
        s = qe @ emb["obj_text_emb"].T
        for r in np.argsort(s)[::-1][:TOP_K]:
            o = objects[r]
            hits.append(mk_hit(o["t0"], o["t1"], s[r], "objects", o["description"]))
        labels = sorted({o["label"] for o in objects})
        le = te.embed(labels)
        ls = qe @ le.T
        for r in np.argsort(ls)[::-1][:TOP_K]:
            lab = labels[r]
            for o in [x for x in objects if x["label"] == lab][:2]:
                hits.append(mk_hit(o["t0"], o["t1"], ls[r], "objects", lab))
    if len(scenes):
        s = qe @ emb["scene_emb"].T
        for r in np.argsort(s)[::-1][:TOP_K]:
            sc = scenes[r]
            hits.append(mk_hit(sc["t0"], sc["t1"], s[r], "scenes", sc["caption"]))
    if len(asr):
        s = qe @ asr_emb.T
        for r in np.argsort(s)[::-1][:TOP_K]:
            a = asr[r]
            hits.append(mk_hit(a["start"], a["end"], s[r], "asr", a["text"]))
    return merge_hits(hits)


def search_image(clip, img_path, objects, scenes, emb):
    hits = []
    q = clip.embed_images([Image.open(img_path).convert("RGB")])[0]
    if len(objects):
        s = q @ emb["obj_emb"].T
        for r in np.argsort(s)[::-1][:TOP_K]:
            o = objects[r]
            hits.append(mk_hit(o["t0"], o["t1"], s[r], "objects", o["description"]))
    if len(scenes):
        s = q @ emb["frame_emb"].T
        for r in np.argsort(s)[::-1][:TOP_K]:
            sc = scenes[r]
            hits.append(mk_hit(sc["t0"], sc["t1"], s[r], "scenes", sc["caption"]))
    return merge_hits(hits)


def overlap_sec(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def hit_ok(h, gt):
    return any(overlap_sec((h["t0"], h["t1"]), g) >= 1.0 for g in gt)


def eval_query(hits, gt):
    if not hits:
        return 0.0, 0.0
    relevant = sum(1 for h in hits if hit_ok(h, gt))
    covered = sum(1 for g in gt
                  if any(overlap_sec((h["t0"], h["t1"]), g) >= 1.0 for h in hits))
    return relevant / len(hits), covered / len(gt) if gt else 0.0


def main():
    _, scenes_tiny, _, _, _ = load_index("tiny")
    _, _, asr_large, _, _ = load_index("large-v3")
    objects, scenes, _, emb, _ = load_index("tiny")
    gt = build_gt(objects, scenes, asr_large)

    te = TextEmbedder()
    clip = CLIPEmbedder()

    results = {}
    for qid, q in gt.items():
        image_hits = None
        if q.get("image"):
            image_hits = search_image(clip, IDX / q["image"], objects, scenes, emb)
        row = {}
        for asr_key in ["tiny", "base", "large-v3"]:
            o2, s2, a2, e2, ae2 = load_index(asr_key)
            channels = []
            if q.get("text"):
                channels.append("text")
            if q.get("image"):
                channels.append("image")
            if q.get("text") and q.get("image"):
                channels.append("fusion")
            for chan in channels:
                if chan == "text":
                    hits = search_text(te, q["text"], o2, s2, a2, e2, ae2)
                elif chan == "fusion":
                    hits = merge_hits(
                        search_text(te, q["text"], o2, s2, a2, e2, ae2)
                        + (image_hits or []))
                else:
                    hits = image_hits
                p, r = eval_query(hits, q["gt"])
                row[f"{chan}|{asr_key}"] = {"p": round(p, 3), "r": round(r, 3),
                                            "n_hits": len(hits),
                                            "n_gt": len(q["gt"])}
        results[qid] = row

    # aggregate
    print(f"{'query':<20} {'channel':<10} {'asr':<8} {'P@5':>6} {'R@5':>6}  gt")
    agg = {}
    for qid, row in results.items():
        for k, v in row.items():
            chan, asr_key = k.split("|")
            key = (chan, asr_key)
            agg.setdefault(key, []).append(v)
            print(f"{qid:<20} {chan:<10} {asr_key:<8} {v['p']:>6.3f} {v['r']:>6.3f}  {v['n_gt']}")
    print("\n== 汇总（macro average）==")
    print(f"{'channel':<10} {'asr':<8} {'avg P@5':>8} {'avg R@5':>8}")
    summary = {}
    for (chan, asr_key), vals in sorted(agg.items()):
        ap = sum(v["p"] for v in vals) / len(vals)
        ar = sum(v["r"] for v in vals) / len(vals)
        summary[f"{chan}|{asr_key}"] = {"avg_p": round(ap, 3), "avg_r": round(ar, 3)}
        print(f"{chan:<10} {asr_key:<8} {ap:>8.3f} {ar:>8.3f}")

    out = Path(__file__).resolve().parent / "eval_results.json"
    out.write_text(json.dumps({"results": results, "summary": summary},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
