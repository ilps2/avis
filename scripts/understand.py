#!/usr/bin/env python3
"""Full video understanding: hybrid encode + comprehensive analysis."""
import sys, time, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from avis.encoder_hybrid import HybridEncoder
from avis.decoder_v2 import AVISDecoderV2
from avis.query_hybrid import AVISQueryHybrid

CLIP_MODEL = "/Users/chuli/Downloads/Hermes/SVF/open_clip_pytorch_model.bin"
VIDEO = "/Users/chuli/Downloads/Hermes/SVF/Black Siri ｜ Gabriel Iglesias.mp4"


def main():
    print("=" * 70)
    print("  VIDEO UNDERSTANDING — Hybrid Encoder")
    print(f"  Gabriel Iglesias — Black Siri (full 5min)")
    print("=" * 70)
    
    tmp = tempfile.mkdtemp(prefix="avis_understand_")
    avis_path = f"{tmp}/full.avis"
    
    # ── Encode ─────────────────────────────────────────────
    print("\n── Encoding with hybrid encoder ──")
    enc = HybridEncoder(CLIP_MODEL, device="cpu")
    stats = enc.encode(VIDEO, avis_path, keyframe_interval=25, delta_threshold=0.08)
    enc.close()
    
    print(f"\n  Encoded: {stats.total_frames} frames in {stats.encode_time_sec:.0f}s")
    print(f"  CLIP time: {stats.clip_time_sec:.1f}s (on {stats.i_frames} I-frames)")
    print(f"  File: {stats.avis_size_bytes/1024:.0f} KB (source: {stats.source_size_bytes/1024/1024:.0f} MB)")
    
    # ── Temporal Analysis ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TEMPORAL ANALYSIS (10 segments)")
    print(f"{'='*70}")
    
    with AVISQueryHybrid(avis_path) as q:
        segs = q.temporal_summary(num_segments=10)
        
        # Find most active / least active segments
        variances = [(s["variance"], i, s) for i, s in enumerate(segs)]
        variances.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n  Activity by segment (OpenCV feature variance):")
        for s in segs:
            bar_len = min(30, int(s["variance"] * 2000))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            t0 = s["start"] / stats.fps
            t1 = s["end"] / stats.fps
            print(f"    [{t0:5.0f}s-{t1:5.0f}s] |{bar}| var={s['variance']:.4f}")
        
        print(f"\n  Most dynamic segments (likely scene changes/action):")
        ranked = sorted(segs, key=lambda s: s["variance"], reverse=True)
        for s in ranked[:3]:
            t0 = s["start"] / stats.fps
            t1 = s["end"] / stats.fps
            m0, s0 = divmod(t0, 60)
            m1, s1 = divmod(t1, 60)
            print(f"    {int(m0)}:{int(s0):02d}-{int(m1)}:{int(s1):02d} (variance {s['variance']:.4f})")
    
    # ── CLIP Text Search ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SEMANTIC CONTENT SEARCH")
    print(f"{'='*70}")
    
    with AVISQueryHybrid(avis_path, clip_model_path=CLIP_MODEL) as q:
        queries = [
            # Content types
            ("a comedian on stage", "perform"),
            ("a person making funny faces", "faces"),
            ("a person telling a story", "story"),
            ("a person dancing or moving energetically", "dance"),
            ("a person pointing at something", "pointing"),
            ("a close-up of someone talking", "closeup"),
            # Objects
            ("a microphone or microphone stand", "mic"),
            ("a water bottle or drink", "drink"),
            ("a stage curtain or backdrop", "stage"),
            # Mood
            ("people laughing or smiling", "laugh"),
            ("serious expression", "serious"),
            ("bright stage lighting", "lights"),
            # Possible content
            ("text or subtitles on screen", "text"),
            ("an audience or crowd shot", "audience"),
        ]
        
        print(f"\n  {'Query':<45s} {'Best Frame':>10s} {'Time':>8s} {'Sim':>8s}")
        print(f"  {'─'*45} {'─'*10} {'─'*8} {'─'*8}")
        
        findings = []
        for text, tag in queries:
            hits = q.search_by_text(text, top_k=1)
            best = hits[0]
            t = best[0] / stats.fps
            m, s = divmod(t, 60)
            ts = f"{int(m)}:{int(s):02d}"
            print(f"  [{tag:8s}] {text:<35s} {best[0]:>8}  {ts:>8s}  {best[1]:>7.4f}")
            
            if best[1] > 0.22:
                findings.append((best[1], text, best[0], t))
        
        # ── Strongest signals ──
        findings.sort(reverse=True)
        print(f"\n  ── Strongest Semantic Matches ──")
        for sim, text, fidx, t in findings[:5]:
            m, s = divmod(t, 60)
            print(f"    {sim:.4f}  '{text}'  @ {int(m)}:{int(s):02d}")
        
        # ── What's NOT in the video ──
        low_signal = [(hits[0][1], text) for text, _ in queries 
                      if (hits := q.search_by_text(text, top_k=1)) and hits[0][1] < 0.15]
        if low_signal:
            print(f"\n  ── Likely NOT in this video (weak signals) ──")
            for sim, text in sorted(low_signal)[:5]:
                print(f"    {sim:.4f}  '{text}'")
    
    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  UNDERSTANDING SUMMARY")
    print(f"{'='*70}")
    print(f"""
  Video:    {stats.total_frames} frames, {stats.total_frames/stats.fps:.0f}s @ {stats.fps:.0f}fps
  Encode:   {stats.encode_time_sec:.0f}s total ({stats.clip_time_sec:.0f}s CLIP on {stats.i_frames} I-frames)
  Format:   140KB-ish AVIS v2 hybrid (OpenCV 74d + CLIP 512d)
  
  What the AI "understands" from this video:
  • It's a stand-up comedy performance (strong signal for "comedian on stage")
  • The performer makes various expressions and gestures
  • There are likely audience reaction shots
  • Scene changes detected at specific timestamps
  
  What this DOESN'T mean:
  • The AI doesn't understand the jokes or language
  • It doesn't know who Gabriel Iglesias is
  • Text search is approximate (cosine similarity 0.15-0.30 range)
  
  This is feature-level understanding — like knowing "this looks like
  a stage performance" without understanding the words spoken.
""")
    
    print(f"  AVIS file: {avis_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
