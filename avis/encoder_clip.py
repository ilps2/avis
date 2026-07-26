"""
AVIS Encoder with CLIP backend (local model file).

CLIP ViT-B-32: 512-dim semantic embeddings.
Requires pre-downloaded open_clip_pytorch_model.bin.
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
    model_name: str


class CLIPExtractor:
    """CLIP ViT-B-32 feature extractor with local model file."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        print(f"  Loading CLIP from {Path(model_path).name}...", end=" ", flush=True)
        t0 = time.time()
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=model_path
        )
        self.model = self.model.to(device).eval()
        self.dim = self.model.visual.output_dim
        
        print(f"done ({time.time()-t0:.1f}s, {self.dim}-dim)")
    
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract CLIP visual embedding for a single BGR frame."""
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(img)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32).squeeze(0)


def encode_video_clip(
    video_path: str,
    output_path: str,
    model_path: str,
    keyframe_interval: int = 30,
    delta_threshold: float = 0.04,
    device: str = "cpu",
    verbose: bool = True,
) -> EncodeStats:
    """
    Encode a video to AVIS using CLIP ViT-B-32 features.
    
    Args:
        video_path: Input video file
        output_path: Output .avis file
        model_path: Path to open_clip_pytorch_model.bin
        keyframe_interval: frames between mandatory I-frames
        delta_threshold: cosine distance threshold for new I-frame
        device: 'cpu' or 'cuda'
        verbose: print progress
    
    Returns:
        EncodeStats
    """
    t0 = time.time()
    
    extractor = CLIPExtractor(model_path, device)
    latent_dim = extractor.dim
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        total_frames = 999999
    
    out = open(output_path, "wb")
    
    model_tag = f"CLIP-ViT-B-32/laion2b".encode("utf-8")[:63].ljust(64, b"\x00")
    header = struct.pack(
        HEADER_FMT,
        MAGIC, VERSION, model_tag,
        latent_dim, keyframe_interval, fps,
        0, width, height,
    )
    out.write(header)
    
    i_count = 0
    d_count = 0
    last_iframe_features = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if width > 640:
            frame = cv2.resize(frame, (640, int(640 * height / width)))
        
        features = extractor.extract(frame)
        
        is_iframe = False
        if last_iframe_features is None:
            is_iframe = True
        elif (frame_idx % keyframe_interval) == 0:
            is_iframe = True
        else:
            cos_dist = 1.0 - float(np.dot(features, last_iframe_features))
            if cos_dist > delta_threshold:
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
            elapsed = time.time() - t0
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx/elapsed:.1f} fps)...")
    
    cap.release()
    
    out.seek(struct.calcsize(">4s I 64s I I f"), 0)
    out.write(struct.pack(">I", frame_idx))
    out.close()
    
    elapsed = time.time() - t0
    avis_size = Path(output_path).stat().st_size
    source_size = Path(video_path).stat().st_size
    total_pixel_bytes = width * height * 3 * frame_idx
    
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
        model_name="CLIP-ViT-B-32",
    )
