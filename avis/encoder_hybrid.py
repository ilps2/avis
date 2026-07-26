"""
AVIS Hybrid Encoder — OpenCV scheduler + CLIP on keyframes.

Strategy:
  - Every frame: OpenCV features (fast, ~0.01ms) → decide I/Δ
  - I-frames ONLY:   CLIP features (~80ms) → semantic annotation
  - Δ-frames:        OpenCV delta only (tiny)
  
Result: CLIP cost on ~10% of frames, but all I-frames have semantic labels.
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

from .extractors import CombinedExtractor
from .format_v2 import (
    MAGIC, FRAME_TYPE_IFRAME, FRAME_TYPE_DELTA,
    HEADER_FMT_V2, HEADER_SIZE_V2,
    FRAME_PREFIX_FMT, FRAME_PREFIX_SIZE,
    DELTA_EXTRA_FMT, DELTA_EXTRA_SIZE,
    IFRAME_EXTRA_FMT, IFRAME_EXTRA_SIZE,
    LAYER_HDR_FMT, LAYER_HDR_SIZE,
    pack_layer, unpack_layer,
    FEATURE_DTYPE_BYTES,
)

VERSION = 2


@dataclass
class HybridStats:
    source_path: str
    source_size_bytes: int
    avis_size_bytes: int
    total_frames: int
    i_frames: int
    delta_frames: int
    opencv_dim: int
    clip_dim: int
    compression_vs_raw: float
    compression_vs_pixels: float
    encode_time_sec: float
    clip_time_sec: float
    fps: float


class HybridEncoder:
    """
    Two-tier encoder:
      Tier 1 (every frame): OpenCV HSV+Edge features → I/Δ scheduling
      Tier 2 (I-frames only): CLIP ViT-B-32 → semantic annotation
    """
    
    def __init__(self, clip_model_path: str, device: str = "cpu"):
        self.device = device
        
        # Tier 1: fast OpenCV extractor
        self.fast_ext = CombinedExtractor()
        self.opencv_dim = self.fast_ext.dim
        
        # Tier 2: CLIP (loaded lazily or eagerly)
        print(f"  Loading CLIP for hybrid encoder...", end=" ", flush=True)
        t0 = time.time()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=clip_model_path
        )
        self.clip_model = self.clip_model.to(device).eval()
        self.clip_dim = self.clip_model.visual.output_dim
        print(f"done ({time.time()-t0:.1f}s)")
        
        self._clip_time_total = 0.0
    
    def _opencv_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.fast_ext.extract(frame_bgr)
    
    def _clip_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract CLIP embedding (expensive, ~80ms)."""
        from PIL import Image
        t0 = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        self._clip_time_total += time.time() - t0
        return feat.cpu().numpy().astype(np.float32).squeeze(0)
    
    def encode(
        self,
        video_path: str,
        output_path: str,
        keyframe_interval: int = 25,
        delta_threshold: float = 0.08,
        verbose: bool = True,
    ) -> HybridStats:
        t_start = time.time()
        self._clip_time_total = 0.0
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames <= 0:
            total_frames = 999999
        
        out = open(output_path, "wb")
        
        # Header (V2)
        model_tag = f"hybrid:opencv{self.opencv_dim}+clip{self.clip_dim}".encode()[:63].ljust(64, b"\x00")
        header = struct.pack(HEADER_FMT_V2, MAGIC, VERSION, model_tag, 0, keyframe_interval, fps, 0, width, height)
        out.write(header)
        
        i_count, d_count = 0, 0
        last_opencv = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if width > 640:
                frame = cv2.resize(frame, (640, int(640 * height / width)))
            
            # ── Tier 1: always extract OpenCV ──
            ocv_feat = self._opencv_features(frame)
            
            # ── Scheduling decision ──
            is_iframe = False
            if last_opencv is None:
                is_iframe = True
            elif (frame_idx % keyframe_interval) == 0:
                is_iframe = True
            else:
                cos_dist = 1.0 - float(np.dot(ocv_feat, last_opencv))
                if cos_dist > delta_threshold:
                    is_iframe = True
            
            ts = frame_idx / fps if fps > 0 else 0.0
            
            if is_iframe:
                # ── Tier 2: CLIP only on I-frames ──
                clip_feat = self._clip_features(frame)
                
                # Write I-frame V2
                out.write(struct.pack(FRAME_PREFIX_FMT, frame_idx, FRAME_TYPE_IFRAME, ts))
                out.write(struct.pack(IFRAME_EXTRA_FMT, 2))  # 2 layers
                
                # Layer 0: OpenCV
                ocv_compressed = zlib.compress(ocv_feat.astype(np.float16).tobytes(), level=6)
                out.write(pack_layer(self.opencv_dim, ocv_compressed))
                
                # Layer 1: CLIP
                clip_compressed = zlib.compress(clip_feat.astype(np.float16).tobytes(), level=6)
                out.write(pack_layer(self.clip_dim, clip_compressed))
                
                last_opencv = ocv_feat
                i_count += 1
            else:
                # ── Δ-frame: OpenCV delta only ──
                delta = ocv_feat - last_opencv
                magnitude = float(np.linalg.norm(delta))
                
                out.write(struct.pack(FRAME_PREFIX_FMT, frame_idx, FRAME_TYPE_DELTA, ts))
                out.write(struct.pack(DELTA_EXTRA_FMT, magnitude, 1))  # 1 layer
                
                delta_compressed = zlib.compress(delta.astype(np.float16).tobytes(), level=6)
                out.write(pack_layer(self.opencv_dim, delta_compressed))
                
                d_count += 1
            
            frame_idx += 1
            if verbose and frame_idx % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  Frame {frame_idx}/{total_frames} "
                      f"({frame_idx/elapsed:.1f} fps, {i_count} I, {d_count} Δ)...")
        
        cap.release()
        
        # Patch total_frames
        out.seek(struct.calcsize(">4s I 64s I I f"), 0)
        out.write(struct.pack(">I", frame_idx))
        out.close()
        
        elapsed = time.time() - t_start
        avis_size = Path(output_path).stat().st_size
        source_size = Path(video_path).stat().st_size
        total_pixel_bytes = width * height * 3 * frame_idx
        
        return HybridStats(
            source_path=video_path,
            source_size_bytes=source_size,
            avis_size_bytes=avis_size,
            total_frames=frame_idx,
            i_frames=i_count,
            delta_frames=d_count,
            opencv_dim=self.opencv_dim,
            clip_dim=self.clip_dim,
            compression_vs_raw=avis_size / max(source_size, 1),
            compression_vs_pixels=avis_size / max(total_pixel_bytes, 1),
            encode_time_sec=elapsed,
            clip_time_sec=self._clip_time_total,
            fps=fps,
        )
    
    def close(self):
        self.clip_model = None
        if self.device != "cpu":
            torch.cuda.empty_cache()
