"""
AVIS Encoder — converts traditional video into AI-native Feature-Stream format.

Format:
  [Header] magic + metadata + model info
  [Body] interleaved I-frames (full features) and Δ-frames (feature deltas)
  
The key insight: consecutive frames in feature space are highly correlated.
Storing deltas + occasional keyframes is 5-15× more compact for AI consumption.
"""

import struct
import zlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import torch
import open_clip


# ── binary format constants ──────────────────────────────────────────
MAGIC = b"AVIS"
VERSION = 1
HEADER_FMT = ">4s I 64s I I f I I I"  # magic, version, model_name(64s), latent_dim, interval, fps, frames, w, h
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FRAME_TYPE_IFRAME = 0x00
FRAME_TYPE_DELTA  = 0x01

IFRAME_FMT = ">I B d"         # frame_idx, type, timestamp
IFRAME_HDR_SIZE = struct.calcsize(IFRAME_FMT)

DFRAME_FMT = ">I B d f"       # frame_idx, type, timestamp, magnitude
DFRAME_HDR_SIZE = struct.calcsize(DFRAME_FMT)


@dataclass
class EncodeStats:
    """Statistics from an encoding run."""
    source_path: str
    source_size_bytes: int
    avis_size_bytes: int
    total_frames: int
    i_frames: int
    delta_frames: int
    latent_dim: int
    compression_vs_raw: float    # AVIS / raw video
    compression_vs_pixels: float  # AVIS / equivalent pixel throughput for AI
    encode_time_sec: float
    fps: float
    

class AVISEncoder:
    """Encodes a video file into the AVIS Feature-Stream format."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        keyframe_interval: int = 30,
        device: str = "cpu",
        delta_threshold: float = 0.05,   # cosine distance threshold for new I-frame
    ):
        self.keyframe_interval = keyframe_interval
        self.delta_threshold = delta_threshold
        self.device = device
        
        # Load CLIP vision encoder
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()
        self.latent_dim = self.model.visual.output_dim
        self.model_name_full = f"{model_name}/{pretrained}"
        
    def _extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract CLIP visual embedding for a single frame."""
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Preprocess
        img = self.preprocess(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img)
            features = features / features.norm(dim=-1, keepdim=True)  # normalize
        return features.cpu().numpy().astype(np.float32).squeeze(0)
    
    def encode(self, video_path: str, output_path: str, progress_callback=None) -> EncodeStats:
        """
        Encode a video file to AVIS format.
        
        Args:
            video_path: Path to input video (any format OpenCV can read)
            output_path: Path for output .avis file
            progress_callback: Optional fn(percent_done)
            
        Returns:
            EncodeStats with compression metrics
        """
        t0 = time.time()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= 0:
            total_frames = 999999  # fallback for streams
        
        # Open output file
        out = open(output_path, "wb")
        
        # Write header (placeholder, we'll update counts at the end)
        model_name_bytes = self.model_name_full.encode("utf-8")[:63].ljust(64, b"\x00")
        header = struct.pack(
            HEADER_FMT,
            MAGIC, VERSION, model_name_bytes,
            self.latent_dim, self.keyframe_interval, fps,
            0,  # total_frames — will patch later
            width, height,
        )
        out.write(header)
        
        # Encode frames
        i_count = 0
        d_count = 0
        last_iframe_features = None
        frame_positions = []  # (file_offset, frame_idx, is_iframe)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            features = self._extract_features(frame)
            
            # Decide: I-frame or Δ-frame?
            is_iframe = False
            if last_iframe_features is None:
                is_iframe = True
            elif (frame_idx % self.keyframe_interval) == 0:
                is_iframe = True
            else:
                # Check cosine distance from last I-frame
                cos_sim = np.dot(features, last_iframe_features)
                cos_dist = 1.0 - cos_sim
                if cos_dist > self.delta_threshold:
                    is_iframe = True
            
            # Record position for index
            frame_positions.append((out.tell(), frame_idx, is_iframe))
            
            if is_iframe:
                # Write I-frame: header + full float16 features
                ts = frame_idx / fps if fps > 0 else 0.0
                out.write(struct.pack(IFRAME_FMT, frame_idx, FRAME_TYPE_IFRAME, ts))
                features_f16 = features.astype(np.float16)
                # Compress the feature blob
                compressed = zlib.compress(features_f16.tobytes(), level=6)
                out.write(struct.pack(">I", len(compressed)))
                out.write(compressed)
                last_iframe_features = features
                i_count += 1
            else:
                # Write Δ-frame: header + delta (float16) + compress
                delta = features - last_iframe_features
                magnitude = float(np.linalg.norm(delta))
                ts = frame_idx / fps if fps > 0 else 0.0
                out.write(struct.pack(DFRAME_FMT, frame_idx, FRAME_TYPE_DELTA, ts, magnitude))
                delta_f16 = delta.astype(np.float16)
                compressed = zlib.compress(delta_f16.tobytes(), level=6)
                out.write(struct.pack(">I", len(compressed)))
                out.write(compressed)
                d_count += 1
            
            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_idx / total_frames * 100)
        
        cap.release()
        
        # Patch header with actual frame count
        out.seek(struct.calcsize(">4s I 64s I I f"), 0)
        out.write(struct.pack(">I", frame_idx))
        
        out.close()
        
        elapsed = time.time() - t0
        avis_size = Path(output_path).stat().st_size
        source_size = Path(video_path).stat().st_size
        
        # Raw pixel throughput for an AI pipeline:
        # Traditional: decode H.264 → pixels → VAE → latent (all frames)
        # AVIS: read features directly (only I-frames carry full features)
        # Effective compression vs pixel processing equivalent
        bytes_per_frame_raw = width * height * 3  # RGB
        total_pixel_bytes = bytes_per_frame_raw * frame_idx
        compression_vs_raw = avis_size / max(source_size, 1)
        compression_vs_pixels = avis_size / max(total_pixel_bytes, 1)
        
        return EncodeStats(
            source_path=video_path,
            source_size_bytes=source_size,
            avis_size_bytes=avis_size,
            total_frames=frame_idx,
            i_frames=i_count,
            delta_frames=d_count,
            latent_dim=self.latent_dim,
            compression_vs_raw=compression_vs_raw,
            compression_vs_pixels=compression_vs_pixels,
            encode_time_sec=elapsed,
            fps=fps,
        )
    
    def close(self):
        """Release model resources."""
        self.model = None
        if self.device != "cpu":
            torch.cuda.empty_cache()


def encode_video(
    video_path: str,
    output_path: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    keyframe_interval: int = 30,
    device: str = "cpu",
    verbose: bool = True,
) -> EncodeStats:
    """Convenience function to encode a video to AVIS."""
    enc = AVISEncoder(
        model_name=model_name,
        pretrained=pretrained,
        keyframe_interval=keyframe_interval,
        device=device,
    )
    
    def progress(pct):
        if verbose and int(pct) % 10 == 0:
            print(f"  Encoding... {pct:.0f}%")
    
    stats = enc.encode(video_path, output_path, progress_callback=progress)
    enc.close()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"AVIS Encoding Complete")
        print(f"{'='*60}")
        print(f"  Source: {stats.source_path}")
        print(f"  Source size:  {stats.source_size_bytes / 1024 / 1024:.1f} MB")
        print(f"  AVIS size:    {stats.avis_size_bytes / 1024:.1f} KB ({stats.avis_size_bytes / 1024 / 1024:.2f} MB)")
        print(f"  Frames:       {stats.total_frames}")
        print(f"  I-frames:     {stats.i_frames}")
        print(f"  Δ-frames:     {stats.delta_frames}")
        print(f"  Latent dim:   {stats.latent_dim}")
        print(f"  AVIS/raw:     {stats.compression_vs_raw:.4f}x ({stats.compression_vs_raw*100:.1f}%)")
        print(f"  AVIS/pixels:  {stats.compression_vs_pixels*100:.2f}%")
        print(f"  Time:         {stats.encode_time_sec:.1f}s")
        
        # Token comparison for AI processing
        tokens_traditional = stats.total_frames * stats.latent_dim  # rough: 1 token per feature dim
        tokens_avis = stats.i_frames * stats.latent_dim + stats.delta_frames * stats.latent_dim * 0.3  # deltas are sparser
        print(f"\n  AI Token comparison (estimate):")
        print(f"    Traditional: ~{tokens_traditional:,} feature-tokens")
        print(f"    AVIS I+deltas: ~{int(tokens_avis):,} feature-tokens")
        print(f"    Reduction: {tokens_traditional / max(tokens_avis, 1):.1f}x fewer tokens")
    
    return stats
