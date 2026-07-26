#!/usr/bin/env python3
"""Stand-up comedy clip: 3-way benchmark with targeted CLIP queries."""
import sys, time, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avis import AVISDecoder, AVISQuery, encode_video_lite, encode_video_torch, encode_video_clip

CLIP_MODEL = "/Users/chuli/Downloads/Hermes/SVF/open_clip_pytorch_model.bin"
VIDEO = "/Users/chuli/avis-prototype/data/standup_clip.mp4"


def analyze(label, stats, avis_path, clip_path=None):
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    src_kb, avis_kb = stats.source_size_bytes/1024, stats.avis_size_bytes/1024
    print(f"  {stats.latent_dim}d  |  {stats.encode_time_sec:.1f}s @ {stats.total_frames/stats.encode_time_sec:.1f}fps  |  {stats.i_frames}/{stats.delta_frames} I/Δ")
    print(f"  {src_kb:.0f}KB → {avis_kb:.0f}KB ({stats.compression_vs_raw*100:.1f}%)")
    
    with AVISDecoder(avis_path) as dec:
        feats = dec.all_features()
        sims = [float(np.dot(feats[i], feats[i+1])) for i in range(len(feats)-1)]
        m, mn, sd = np.mean(sims), np.min(sims), np.std(sims)
        changes = dec.scene_changes(threshold=0.04)
    
    print(f"  Frame sim: μ={m:.4f}  min={mn:.4f}  σ={sd:.4f}")
    if changes:
        # Group nearby changes
        groups = []
        for c in changes:
            if groups and c - groups[-1][-1] <= 5:
                groups[-1].append(c)
            else:
                groups.append([c])
        print(f"  Scene changes: {len(changes)} raw → {len(groups)} groups")
        for g in groups[:6]:
            rng = f"{g[0]}-{g[-1]}" if len(g) > 1 else str(g[0])
            t0, t1 = g[0]/stats.fps, g[-1]/stats.fps
            print(f"    frames {rng} ({t0:.1f}s-{t1:.1f}s)")
    else:
        print(f"  Scene changes: none")
    
    dim = stats.latent_dim
    trad = stats.total_frames * dim
    avis_tok = stats.i_frames * dim + int(stats.delta_frames * dim * 0.10)
    print(f"  Tokens: {trad:,} → {avis_tok:,} ({trad/avis_tok:.1f}×)")
    
    return {"sim_mean": m, "sim_min": mn, "sim_std": sd, "changes": changes, "tok_red": trad/avis_tok}


def main():
    print("=" * 72)
    print("  AVIS — Stand-up Comedy Benchmark")
    print(f"  Gabriel Iglesias — 'Black Siri' (25s clip, 635 frames)")
    print("=" * 72)
    
    tmp = tempfile.mkdtemp(prefix="avis_standup_")
    results = {}
    
    # ── Encode ─────────────────────────────────────────────
    print("\n── OpenCV ──")
    s1 = encode_video_lite(VIDEO, f"{tmp}/cv.avis", keyframe_interval=25, delta_threshold=0.08)
    results["cv"] = analyze("OpenCV (74d)", s1, f"{tmp}/cv.avis")
    
    print("\n── PyTorch MobileNetV3 ──")
    s2 = encode_video_torch(VIDEO, f"{tmp}/pt.avis", keyframe_interval=25, delta_threshold=0.05)
    results["pt"] = analyze("PyTorch (576d)", s2, f"{tmp}/pt.avis")
    
    print("\n── CLIP ViT-B-32 ──")
    s3 = encode_video_clip(VIDEO, f"{tmp}/clip.avis", model_path=CLIP_MODEL,
                           keyframe_interval=25, delta_threshold=0.04)
    results["clip"] = analyze("CLIP (512d)", s3, f"{tmp}/clip.avis", clip_path=CLIP_MODEL)
    
    # ── CLIP Text Search ───────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  CLIP TEXT SEARCH — Semantic Frame Retrieval")
    print(f"{'='*72}")
    with AVISQuery(f"{tmp}/clip.avis") as q:
        queries = [
            ("a comedian on stage with a microphone", "stand-up shot"),
            ("a person wearing a black shirt", "outfit match"),
            ("the audience or crowd", "audience reaction"),
            ("a close-up of a face", "close-up"),
            ("a person laughing", "laughter"),
            ("a stage with lights", "stage setup"),
            ("a person making a gesture with hands", "gesture"),
            ("a man telling a joke", "performing"),
        ]
        for text, tag in queries:
            hits = q.search_by_text(text, top_k=3, clip_model_path=CLIP_MODEL)
            best = hits[0]
            print(f"  [{tag:16s}] '{text[:50]}'")
            print(f"              → frame {best[0]:4d} ({best[0]/s3.fps:.1f}s)  sim={best[1]:.4f}")
    
    # ── Comparison Table ────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Metric':<28} {'OpenCV 74d':>14} {'PyTorch 576d':>14} {'CLIP 512d':>14}")
    print(f"  {'─'*28} {'─'*14} {'─'*14} {'─'*14}")
    keys = [
        ("Encode speed",         "encode_fps",     ".1f fps"),
        ("AVIS size",            "avis_kb",        ".0f KB"),
        ("Frame sim mean",       "sim_mean",       ".4f"),
        ("Frame sim min",        "sim_min",        ".4f"),
        ("Token reduction",      "tok_red",        ".1f×"),
    ]
    stats_map = {"cv": s1, "pt": s2, "clip": s3}
    for label, key, fmt in keys:
        if key in ("sim_mean", "sim_min", "tok_red"):
            v = {bk: results[bk][key] for bk in ["cv", "pt", "clip"]}
        elif key == "encode_fps":
            v = {bk: stats_map[bk].total_frames / stats_map[bk].encode_time_sec for bk in ["cv", "pt", "clip"]}
        elif key == "avis_kb":
            v = {bk: stats_map[bk].avis_size_bytes / 1024 for bk in ["cv", "pt", "clip"]}
        else:
            continue
        
        if fmt == ".1f fps":
            print(f"  {label:<28} {v['cv']:>13.1f}fps {v['pt']:>13.1f}fps {v['clip']:>13.1f}fps")
        elif fmt == ".0f KB":
            print(f"  {label:<28} {v['cv']:>13.0f}KB {v['pt']:>13.0f}KB {v['clip']:>13.0f}KB")
        else:
            print(f"  {label:<28} {v['cv']:>14.4f} {v['pt']:>14.4f} {v['clip']:>14.4f}")
    
    print(f"\n  Files: {tmp}/{{cv,pt,clip}}.avis")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
