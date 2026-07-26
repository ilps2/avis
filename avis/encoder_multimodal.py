"""
AVIS Multimodal Encoder — Hybrid visual + Whisper audio transcript.

Pipeline:
  1. Extract audio → transcribe with Whisper
  2. Encode visual features with HybridEncoder (OpenCV + CLIP)
  3. Append transcript block to AVIS file
"""

import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import whisper

from .encoder_hybrid import HybridEncoder, HybridStats
from .format_v2 import MAGIC, HEADER_FMT_V2
from .format_v3 import VERSION_V3, write_transcript_block, Transcript


@dataclass
class MultimodalStats:
    visual: HybridStats
    transcript_segments: int
    transcript_chars: int
    whisper_time_sec: float
    whisper_model: str


def encode_multimodal(
    video_path: str,
    output_path: str,
    clip_model_path: str,
    whisper_model_name: str = "base",
    keyframe_interval: int = 25,
    delta_threshold: float = 0.08,
    language: str = "en",
    device: str = "cpu",
    verbose: bool = True,
) -> MultimodalStats:
    """
    Encode video to AVIS V3: hybrid visual features + Whisper transcript.
    
    Args:
        video_path: Input video
        output_path: Output .avis file
        clip_model_path: CLIP model for visual encoding
        whisper_model_name: 'tiny', 'base', 'small', 'medium', 'large'
        keyframe_interval: Visual I-frame interval
        delta_threshold: Visual delta threshold
        language: Whisper language hint
        device: 'cpu' or 'cuda'
        verbose: Print progress
    
    Returns:
        MultimodalStats
    """
    
    # ── Step 1: Extract audio ──────────────────────────────
    if verbose:
        print("  Extracting audio...", end=" ", flush=True)
    
    audio_path = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path,
    ], capture_output=True, check=True)
    
    if verbose:
        print("done")
    
    # ── Step 2: Transcribe ─────────────────────────────────
    if verbose:
        print(f"  Transcribing with Whisper {whisper_model_name}...", end=" ", flush=True)
    
    t0 = time.time()
    wmodel = whisper.load_model(whisper_model_name)
    wresult = wmodel.transcribe(audio_path, language=language)
    whisper_time = time.time() - t0
    
    transcript = Transcript.from_whisper(wresult, f"whisper-{whisper_model_name}")
    
    if verbose:
        print(f"done ({whisper_time:.1f}s, {len(transcript.segments)} segments)")
    
    # ── Step 3: Visual encode ──────────────────────────────
    if verbose:
        print(f"  Encoding visual features...")
    
    enc = HybridEncoder(clip_model_path, device=device)
    vstats = enc.encode(
        video_path, output_path,
        keyframe_interval=keyframe_interval,
        delta_threshold=delta_threshold,
        verbose=verbose,
    )
    enc.close()
    
    # ── Step 4: Append transcript ──────────────────────────
    if verbose:
        print(f"  Appending transcript block...", end=" ", flush=True)
    
    # Update version in header to V3
    with open(output_path, "r+b") as fh:
        fh.seek(4)  # skip magic
        fh.write(struct.pack(">I", VERSION_V3))
    
    # Append transcript at end
    with open(output_path, "ab") as fh:
        write_transcript_block(fh, transcript)
    
    if verbose:
        print(f"done ({len(transcript.full_text)} chars)")
    
    return MultimodalStats(
        visual=vstats,
        transcript_segments=len(transcript.segments),
        transcript_chars=len(transcript.full_text),
        whisper_time_sec=whisper_time,
        whisper_model=whisper_model_name,
    )
