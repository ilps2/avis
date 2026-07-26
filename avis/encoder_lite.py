"""
AVIS Lightweight Encoder — uses OpenCV features, no CLIP/GPU needed.

Produces the same binary AVIS format as the CLIP encoder.
"""

import struct
import zlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from .extractors import CombinedExtractor


MAGIC = b"AVIS"
VERSION = 1
HEADER_FMT = ">4s I 64s I I f I I I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FRAME_TYPE_IFRAME = 0x00
FRAME_TYPE_DELTA  = 0x01

IFRAME_FMT = ">I B d"
IFRAME_HDR_SIZE = struct.calcsize(IFRAME_FMT)
DFRAME_FMT = ">I B d f"
DFRAME_HDR_SIZE = struct.calcsize(DFRAME_FMT)


@dataclass
class EncodeStats:
    source_path: str
    source_size_bytes: int
    avis_size_bytes: int
    total_frames: int
    i_frames: int
    delta_frames: int
    latent_dim: int
    compression_vs_raw: float
    compression_vs_pixels: float
    encode_time_sec: float
    fps: float


def encode_video_lite(
    video_path: str,
    output_path: str,
    keyframe_interval: int = 30,
    delta_threshold: float = 0.08,
    verbose: bool = True,
) -> EncodeStats:
    """
    Encode a video to AVIS format using lightweight OpenCV features.
    """
    t0 = time.time()
    
    extractor = CombinedExtractor()
    latent_dim = extractor.dim
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        total_frames = 999999
    
    out = open(output_path, "wb")
    model_name = "OpenCV-HSV+Edge/200d".encode("utf-8")[:63].ljust(64, b"\x00")
    header = struct.pack(HEADER_FMT, MAGIC, VERSION, model_name, latent_dim, keyframe_interval, fps, 0, width, height)
    out.write(header)
    
    i_count = 0
    d_count = 0
    last_iframe_features = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downsample large frames for speed
        if width > 640:
            frame = cv2.resize(frame, (640, int(640 * height / width)))
        
        features = extractor.extract(frame)
        
        is_iframe = False
        if last_iframe_features is None:
            is_iframe = True
        elif (frame_idx % keyframe_interval) == 0:
            is_iframe = True
        else:
            cos_sim = float(np.dot(features, last_iframe_features))
            if (1.0 - cos_sim) > delta_threshold:
                is_iframe = True
        
        if is_iframe:
            ts = frame_idx / fps if fps > 0 else 0.0
            out.write(struct.pack(IFRAME_FMT, frame_idx, FRAME_TYPE_IFRAME, ts))
            features_f16 = features.astype(np.float16)
            compressed = zlib.compress(features_f16.tobytes(), level=6)
            out.write(struct.pack(">I", len(compressed)))
            out.write(compressed)
            last_iframe_features = features
            i_count += 1
        else:
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
        if verbose and frame_idx % 50 == 0:
            print(f"  Encoded {frame_idx} frames...")
    
    cap.release()
    
    # Patch total_frames
    out.seek(struct.calcsize(">4s I 64s I I f"), 0)
    out.write(struct.pack(">I", frame_idx))
    out.close()
    
    elapsed = time.time() - t0
    avis_size = Path(output_path).stat().st_size
    source_size = Path(video_path).stat().st_size
    bytes_per_frame_raw = width * height * 3
    total_pixel_bytes = bytes_per_frame_raw * frame_idx
    
    return EncodeStats(
        source_path=video_path,
        source_size_bytes=source_size,
        avis_size_bytes=avis_size,
        total_frames=frame_idx,
        i_frames=i_count,
        delta_frames=d_count,
        latent_dim=latent_dim,
        compression_vs_raw=avis_size / max(source_size, 1),
        compression_vs_pixels=avis_size / max(total_pixel_bytes, 1),
        encode_time_sec=elapsed,
        fps=fps,
    )
