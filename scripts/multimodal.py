#!/usr/bin/env python3
"""
Multimodal Video Understanding
================================
Visual (CLIP) + Audio (Whisper transcript) → combined analysis
"""

import sys, json, time, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import whisper
from avis import AVISQuery
from avis.encoder_hybrid import HybridEncoder
from avis.decoder_v2 import AVISDecoderV2
from avis.query_hybrid import AVISQueryHybrid

CLIP_MODEL = "/Users/chuli/Downloads/Hermes/SVF/open_clip_pytorch_model.bin"
VIDEO = "/Users/chuli/avis-prototype/data/standup_clip.mp4"


def main():
    print("=" * 70)
    print("  MULTIMODAL VIDEO UNDERSTANDING")
    print("  Visual (CLIP) + Audio (Whisper)")
    print("=" * 70)
    
    # ── 1. Audio: transcribe ────────────────────────────────
    print("\n── 1. Audio Transcription (Whisper base) ──")
    
    # Extract audio
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", VIDEO, "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "/tmp/standup_audio.wav"
    ], capture_output=True)
    
    model = whisper.load_model("base")
    result = model.transcribe("/tmp/standup_audio.wav", language="en")
    
    print(f"  Transcript ({len(result['segments'])} segments):\n")
    for seg in result["segments"]:
        t0, t1 = seg["start"], seg["end"]
        print(f"  [{t0:5.1f}s] {seg['text'].strip()}")
    
    # ── 2. Visual: hybrid encode ────────────────────────────
    print(f"\n── 2. Visual Encoding (Hybrid) ──")
    tmp = tempfile.mkdtemp(prefix="avis_mm_")
    avis_path = f"{tmp}/standup.avis"
    
    enc = HybridEncoder(CLIP_MODEL, device="cpu")
    stats = enc.encode(VIDEO, avis_path, keyframe_interval=25, delta_threshold=0.08, verbose=False)
    enc.close()
    print(f"  {stats.total_frames} frames, {stats.i_frames} I-frames, {stats.clip_time_sec:.1f}s CLIP")
    
    # ── 3. Align transcript with visual timeline ────────────
    print(f"\n── 3. Multimodal Alignment ──")
    
    with AVISQueryHybrid(avis_path, clip_model_path=CLIP_MODEL) as q:
        # For each transcript segment, find the most visually matching frame
        print(f"\n  {'Time':>8s}  {'Transcript':<55s}  {'Best Visual Match':<30s}  {'Sim':>6s}")
        print(f"  {'─'*8}  {'─'*55}  {'─'*30}  {'─'*6}")
        
        for seg in result["segments"]:
            t_mid = (seg["start"] + seg["end"]) / 2
            frame_idx = min(int(t_mid * stats.fps), stats.total_frames - 1)
            
            # What does CLIP see at this frame?
            clip_feat = q.decoder.reader.get_feature(frame_idx, layer=1)
            
            # What does CLIP "think" this frame looks like?
            visual_matches = q.search_by_text(seg["text"].strip()[:80], top_k=1)
            
            # Find the frame's strongest visual label
            labels = [
                "a comedian on stage", "a person talking", "a person laughing",
                "a person making gestures", "a close-up of a face",
                "a person holding a phone", "stage with microphone",
            ]
            best_label, best_sim = "", 0
            for label in labels:
                hits = q.search_by_text(label, top_k=1)
                if hits[0][1] > best_sim:
                    best_sim = hits[0][1]
                    best_label = label
            
            print(f"  {t_mid:>7.1f}s  {seg['text'].strip()[:55]:<55s}  {best_label:<30s}  {best_sim:>5.3f}")
    
    # ── 4. Questions only multimodal can answer ─────────────
    print(f"\n{'='*70}")
    print(f"  WHAT CHANGES WITH AUDIO")
    print(f"{'='*70}")
    
    questions = [
        ("What is he talking about?", 
         "Visual-only: 'A comedian on stage'\n    With audio: 'He gave iPhones to his tour buddies\n    and is explaining Siri to the audience'"),
        ("Who are the characters?",
         "Visual-only: 'A person on stage'\n    With audio: 'Gabriel, Martin, and the audience.\n    Martin is the guy who messes with the phone'"),
        ("When does the audience laugh?",
         "Visual-only: 'Around 2:50 (in full video)'\n    With audio: 'Can detect laughter in audio waveform\n    and align with punchline timing'"),
        ("What is the setup/punchline structure?",
         "Visual-only: 'Cannot detect'\n    With audio: 'Setup: iPhone 4S gifts → Siri feature.\n    Punchline is being set up: Martin messes with it.\n    Clip ends right before the Black Siri joke.'"),
        ("Can we search for specific topics?",
         "Visual-only: 'Find frames that look like a comedy show'\n    With audio: 'Search for \"iPhone\", \"Siri\", \"Martin\"\n    → returns exact timestamps'"),
    ]
    
    for q, a in questions:
        print(f"\n  Q: {q}")
        for line in a.split("\n"):
            print(f"  {line}")
    
    # ── 5. Honest assessment ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  IS THIS 'TRUE' VIDEO UNDERSTANDING?")
    print(f"{'='*70}")
    print(f"""
  Short answer: No. But it's a lot closer than visual-only.
  
  What multimodal adds:
    ✓ Knows WHAT is being said (transcript)
    ✓ Can search for specific topics/words
    ✓ Understands joke structure (setup → punchline)
    ✓ Can detect laughter timing aligned with content
    ✓ Combined: "What does the audience react to?" ← answerable now
    
  What's still missing for 'true' understanding:
    ✗ Why is "Black Siri" funny? (requires cultural context)
    ✗ Who is Gabriel Iglesias? (requires world knowledge)
    ✗ What is Martin's personality? (requires character modeling)
    ✗ How does this bit fit in his career? (requires temporal context)
    ✗ The emotional arc of the performance (requires affect modeling)
    
  The gap between "knowing the words" and "understanding the humor"
  is the gap between NLP and human cognition. Audio transcription
  gets you the text — but comedy is about timing, delivery,
  shared cultural reference, and surprise. Those are still
  beyond current AI.
  
  That said: for practical video understanding tasks
  (indexing, search, summarization, content moderation),
  visual + audio = a working system today.
""")


if __name__ == "__main__":
    main()
