#!/usr/bin/env python3
"""
Hybrid Encoder Benchmark
=========================
Pure OpenCV vs Pure CLIP vs Hybrid (OpenCV scheduler + CLIP on I-frames)
Same standup comedy clip, head-to-head comparison.
"""

import sys, time, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avis import AVISDecoder, AVISQuery, encode_video_lite, encode_video_clip
from avis.encoder_hybrid import HybridEncoder
from avis.decoder_v2 import AVISDecoderV2
from avis.query_hybrid import AVISQueryHybrid

CLIP_MODEL = "/Users/chuli/Downloads/Hermes/SVF/open_clip_pytorch_model.bin"
VIDEO = "/Users/chuli/avis-prototype/data/standup_clip.mp4"


def main():
    print("=" * 72)
    print("  HYBRID ENCODER BENCHMARK")
    print("  Stand-up comedy (635 frames, 25s)")
    print("=" * 72)
    
    tmp = tempfile.mkdtemp(prefix="avis_hybrid_")
    
    # ── 1. Pure OpenCV ─────────────────────────────────────
    print("\n── 1. Pure OpenCV ──")
    t0 = time.time()
    s_ocv = encode_video_lite(VIDEO, f"{tmp}/ocv.avis", keyframe_interval=25, delta_threshold=0.08)
    
    # ── 2. Pure CLIP ───────────────────────────────────────
    print("\n── 2. Pure CLIP ──")
    s_clip = encode_video_clip(VIDEO, f"{tmp}/clip.avis", model_path=CLIP_MODEL,
                                keyframe_interval=25, delta_threshold=0.04)
    
    # ── 3. Hybrid ──────────────────────────────────────────
    print("\n── 3. Hybrid (OpenCV scheduler + CLIP on I-frames) ──")
    hyb = HybridEncoder(CLIP_MODEL, device="cpu")
    s_hyb = hyb.encode(VIDEO, f"{tmp}/hyb.avis", keyframe_interval=25, delta_threshold=0.08)
    hyb.close()
    
    # ── Comparison Table ────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  ENCODING COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Metric':<30} {'OpenCV':>13} {'CLIP':>13} {'Hybrid':>13}")
    print(f"  {'─'*30} {'─'*13} {'─'*13} {'─'*13}")
    
    rows = [
        ("Encode time", 
         f"{s_ocv.encode_time_sec:>12.1f}s", f"{s_clip.encode_time_sec:>12.1f}s", f"{s_hyb.encode_time_sec:>12.1f}s"),
        ("CLIP inference time",
         f"{'n/a':>13}", f"{s_clip.encode_time_sec:>12.1f}s", f"{s_hyb.clip_time_sec:>12.1f}s"),
        ("I-frames / Δ-frames",
         f"{s_ocv.i_frames:>5}/{s_ocv.delta_frames:<5}", f"{s_clip.i_frames:>5}/{s_clip.delta_frames:<5}", f"{s_hyb.i_frames:>5}/{s_hyb.delta_frames:<5}"),
        ("AVIS file size",
         f"{s_ocv.avis_size_bytes/1024:>11.0f}KB", f"{s_clip.avis_size_bytes/1024:>11.0f}KB", f"{s_hyb.avis_size_bytes/1024:>11.0f}KB"),
        ("vs source",
         f"{s_ocv.compression_vs_raw*100:>11.1f}%", f"{s_clip.compression_vs_raw*100:>11.1f}%", f"{s_hyb.compression_vs_raw*100:>11.1f}%"),
    ]
    for label, v1, v2, v3 in rows:
        print(f"  {label:<30} {v1:>13} {v2:>13} {v3:>13}")
    
    # Token economics
    print(f"\n  ── Token Economics ──")
    for name, s in [("OpenCV", s_ocv), ("CLIP", s_clip), ("Hybrid", s_hyb)]:
        dim = s.latent_dim if hasattr(s, 'latent_dim') else s_hyb.opencv_dim
        trad = s.total_frames * (dim if name != "Hybrid" else s_hyb.clip_dim)
        if name == "Hybrid":
            # Hybrid: I-frames carry CLIP (512d), Δ-frames carry OpenCV delta (74d)
            avis_tok = s.i_frames * s_hyb.clip_dim + int(s.delta_frames * s_hyb.opencv_dim * 0.10)
        else:
            avis_tok = s.i_frames * dim + int(s.delta_frames * dim * 0.10)
        print(f"    {name:8s}: {trad:>8,} → {avis_tok:>8,} effective tokens ({trad/avis_tok:.1f}×)")

    # ── Decode & Validate Hybrid ────────────────────────────
    print(f"\n{'='*72}")
    print(f"  HYBRID DECODER VALIDATION")
    print(f"{'='*72}")
    
    with AVISDecoderV2(f"{tmp}/hyb.avis") as dec:
        print(f"  Version: {dec.header.version}")
        print(f"  Model:   {dec.header.model_name}")
        print(f"  Hybrid:  {dec.header.is_hybrid}")
        print(f"  Layers:  OpenCV={dec.header.opencv_dim}d + CLIP={dec.header.clip_dim}d")
        print(f"  I-frames with CLIP: {len(dec.i_frame_indices)}")
        
        # Validate base features
        ocv = dec.opencv_features()
        clip = dec.clip_features()
        print(f"  OpenCV features: {len(ocv)} frames × {ocv[0].shape[0]}d")
        print(f"  CLIP features:   {len(clip)} frames × {clip[0].shape[0]}d")
        
        # Check base layer quality
        sims = [float(np.dot(ocv[i], ocv[i+1])) for i in range(len(ocv)-1)]
        print(f"  OpenCV stability: μ={np.mean(sims):.4f} min={np.min(sims):.4f}")
        
        # Scene detection
        changes = dec.scene_changes(threshold=0.06)
        print(f"  Scene changes: {len(changes)}")
    
    # ── Hybrid Text Search ──────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  HYBRID TEXT SEARCH (CLIP features from I-frames)")
    print(f"{'='*72}")
    
    with AVISQueryHybrid(f"{tmp}/hyb.avis", clip_model_path=CLIP_MODEL) as q:
        queries = [
            "a comedian on stage with a microphone",
            "a person wearing a black shirt",
            "a person laughing",
            "the audience or crowd",
        ]
        for text in queries:
            hits = q.search_by_text(text, top_k=3)
            best = hits[0]
            t = best[0] / s_hyb.fps
            print(f"  '{text[:50]}'")
            print(f"    → frame {best[0]:4d} ({t:.1f}s)  sim={best[1]:.4f}")
    
    # Also search with pure CLIP for comparison
    print(f"\n  Pure CLIP reference (from pure CLIP encode):")
    with AVISQuery(f"{tmp}/clip.avis") as q:
        for text in queries:
            hits = q.search_by_text(text, top_k=3, clip_model_path=CLIP_MODEL)
            best = hits[0]
            t = best[0] / s_clip.fps
            print(f"  '{text[:50]}'")
            print(f"    → frame {best[0]:4d} ({t:.1f}s)  sim={best[1]:.4f}")
    
    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  KEY TAKEAWAYS")
    print(f"{'='*72}")
    
    clip_pct = s_hyb.clip_time_sec / s_clip.encode_time_sec * 100
    size_pct = s_hyb.avis_size_bytes / s_clip.avis_size_bytes * 100
    i_pct = s_hyb.i_frames / s_clip.total_frames * 100
    
    print(f"""
  CLIP computation: {s_hyb.clip_time_sec:.1f}s vs {s_clip.encode_time_sec:.1f}s pure
    → Hybrid uses {clip_pct:.0f}% of the CLIP compute budget
  
  I-frame CLIP coverage: {s_hyb.i_frames} I-frames → {i_pct:.0f}% of video has CLIP annotation
    → Text search works on ALL frames (interpolated from nearest I-frame)
  
  File size: {s_hyb.avis_size_bytes/1024:.0f} KB hybrid vs {s_clip.avis_size_bytes/1024:.0f} KB pure CLIP
    → {size_pct:.0f}% of pure CLIP size
  
  The hybrid approach gives you:
    ✓ Fast scene detection (OpenCV, 80fps tier)
    ✓ Semantic text search (CLIP, on I-frames)
    ✓ Good token compression (Δ-frames carry only lightweight OpenCV deltas)
    ✓ Graceful degradation (if CLIP is slow/unavailable, fall back to OpenCV)
""")
    
    print(f"  Files: {tmp}/{{ocv,clip,hyb}}.avis")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
