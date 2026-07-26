#!/usr/bin/env python3
"""Real-world video: 3-way backend comparison."""
import sys, time, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avis import AVISDecoder, AVISQuery, encode_video_lite, encode_video_torch, encode_video_clip

CLIP_MODEL = "/Users/chuli/Downloads/Hermes/SVF/open_clip_pytorch_model.bin"
VIDEO = "/Users/chuli/Downloads/bilibili_downloads/P01_mmexport1785040001217.mp4"

def analyze(label, stats, avis_path, clip_path=None):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    
    src_kb = stats.source_size_bytes / 1024
    avis_kb = stats.avis_size_bytes / 1024
    
    print(f"  Feature:      {stats.latent_dim}d")
    print(f"  Encode:       {stats.encode_time_sec:.1f}s @ {stats.total_frames/stats.encode_time_sec:.1f} fps")
    print(f"  I/Δ:          {stats.i_frames}/{stats.delta_frames}")
    print(f"  File:         {src_kb:.0f} KB → {avis_kb:.0f} KB ({stats.compression_vs_raw*100:.1f}%)")
    
    with AVISDecoder(avis_path) as dec:
        feats = dec.all_features()
        
        # Frame-to-frame stability
        sims = [float(np.dot(feats[i], feats[i+1])) for i in range(len(feats)-1)]
        print(f"  Consecutive:  mean={np.mean(sims):.4f}  min={np.min(sims):.4f}  std={np.std(sims):.4f}")
        
        # Scene changes
        changes = dec.scene_changes(threshold=0.05)
        if changes:
            print(f"  Scene changes: {len(changes)} at {[f'{c}({c/stats.fps:.1f}s)' for c in changes[:8]]}")
        else:
            print(f"  Scene changes: none detected")
    
    # Token
    dim = stats.latent_dim
    trad = stats.total_frames * dim
    avis_tok = stats.i_frames * dim + int(stats.delta_frames * dim * 0.10)
    print(f"  Tokens:       {trad:,} → {avis_tok:,} ({trad/avis_tok:.1f}×)")
    
    return {"sim_mean": np.mean(sims), "sim_min": np.min(sims), "changes": changes}


def main():
    print("=" * 72)
    print("  AVIS — Real Video Benchmark")
    print(f"  Source: {Path(VIDEO).name}")
    print("=" * 72)
    
    tmp = tempfile.mkdtemp(prefix="avis_real_")
    results = {}
    
    # OpenCV
    print("\n── OpenCV ──")
    s1 = encode_video_lite(VIDEO, f"{tmp}/cv.avis", keyframe_interval=30, delta_threshold=0.08)
    results["cv"] = analyze("OpenCV (74d)", s1, f"{tmp}/cv.avis")
    
    # PyTorch
    print("\n── PyTorch ──")
    s2 = encode_video_torch(VIDEO, f"{tmp}/pt.avis", keyframe_interval=30, delta_threshold=0.05)
    results["pt"] = analyze("PyTorch (576d)", s2, f"{tmp}/pt.avis")
    
    # CLIP
    print("\n── CLIP ──")
    s3 = encode_video_clip(VIDEO, f"{tmp}/clip.avis", model_path=CLIP_MODEL,
                           keyframe_interval=30, delta_threshold=0.03)
    results["clip"] = analyze("CLIP (512d)", s3, f"{tmp}/clip.avis", clip_path=CLIP_MODEL)
    
    # CLIP Text Search
    print(f"\n{'='*72}")
    print(f"  CLIP TEXT SEARCH (killer feature)")
    print(f"{'='*72}")
    with AVISQuery(f"{tmp}/clip.avis") as q:
        queries = [
            "a person",
            "food or cooking",
            "outdoor scene",
            "text or writing on screen",
            "close-up of a face",
            "a building or city",
            "nature or trees",
            "a cartoon or animation",
        ]
        for text in queries:
            hits = q.search_by_text(text, top_k=3, clip_model_path=CLIP_MODEL)
            best = hits[0]
            t = best[0] / s3.fps
            print(f"  '{text}': top={best[0]}({t:.1f}s) sim={best[1]:.4f}")
    
    # Comparison table
    print(f"\n{'='*72}")
    print(f"  COMPARISON")
    print(f"{'='*72}")
    print(f"  {'':<25} {'OpenCV':>12} {'PyTorch':>12} {'CLIP':>12}")
    for label, key in [("Consecutive sim mean", "sim_mean"), ("Consecutive sim min", "sim_min")]:
        print(f"  {label:<25} {results['cv'][key]:>12.4f} {results['pt'][key]:>12.4f} {results['clip'][key]:>12.4f}")
    for bk, s in [("cv", s1), ("pt", s2), ("clip", s3)]:
        print(f"  {bk:>25} encode: {s.encode_time_sec:>6.1f}s  file: {s.avis_size_bytes/1024:>6.0f}KB  tok: {s.total_frames*s.latent_dim/(s.i_frames*s.latent_dim+int(s.delta_frames*s.latent_dim*0.10)):>4.1f}×")


if __name__ == "__main__":
    main()
