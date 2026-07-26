"""
AVIS V2 Decoder — multi-layer frame reader.

Handles both V1 (single-layer) and V2 (multi-layer) formats.
For hybrid files: get_opencv_feature() always available,
get_clip_feature() available for I-frames, interpolated for Δ-frames.
"""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .format_v2 import (
    MAGIC, FRAME_TYPE_IFRAME, FRAME_TYPE_DELTA,
    HEADER_FMT_V1, HEADER_SIZE_V1,
    HEADER_FMT_V2, HEADER_SIZE_V2,
    FRAME_PREFIX_FMT, FRAME_PREFIX_SIZE,
    DELTA_EXTRA_FMT, DELTA_EXTRA_SIZE,
    IFRAME_EXTRA_FMT, IFRAME_EXTRA_SIZE,
    LAYER_HDR_FMT, LAYER_HDR_SIZE,
    pack_layer, unpack_layer,
    FEATURE_DTYPE_BYTES,
)


@dataclass
class AVISHeaderV2:
    version: int
    model_name: str
    is_hybrid: bool
    keyframe_interval: int
    fps: float
    total_frames: int
    width: int
    height: int
    # Parsed from model_name for hybrid
    opencv_dim: int = 0
    clip_dim: int = 0


@dataclass
class FrameEntryV2:
    frame_idx: int
    timestamp: float
    is_iframe: bool
    file_offset: int
    magnitude: float
    num_layers: int
    # Per-layer: (dim, comp_len, data_offset)
    layers: list[tuple[int, int, int]]


class AVISReaderV2:
    """Reads V1 and V2 AVIS files. Multi-layer aware."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._fh = open(path, "rb")
        
        # Detect version
        magic_and_ver = self._fh.read(8)
        self._fh.seek(0)
        magic, version = struct.unpack(">4s I", magic_and_ver)
        
        if magic != MAGIC:
            raise ValueError(f"Not an AVIS file: {magic}")
        
        self._version = version
        self._is_v2 = version >= 2
        
        if self._is_v2:
            self._parse_header_v2()
            self._scan_index_v2()
        else:
            self._parse_header_v1()
            self._scan_index_v1()
    
    def _parse_header_v2(self):
        raw = self._fh.read(HEADER_SIZE_V2)
        fields = struct.unpack(HEADER_FMT_V2, raw)
        model_name = fields[2].decode("utf-8").rstrip("\x00")
        
        # Parse hybrid dimensions from model_name
        is_hybrid = model_name.startswith("hybrid:")
        opencv_dim, clip_dim = 0, 0
        if is_hybrid:
            # "hybrid:opencv74+clip512" or similar
            parts = model_name.replace("hybrid:", "")
            for part in parts.split("+"):
                if "opencv" in part:
                    opencv_dim = int(''.join(c for c in part if c.isdigit()))
                elif "clip" in part:
                    clip_dim = int(''.join(c for c in part if c.isdigit()))
        
        self.header = AVISHeaderV2(
            version=fields[1],
            model_name=model_name,
            is_hybrid=is_hybrid,
            keyframe_interval=fields[4],
            fps=fields[5],
            total_frames=fields[6],
            width=fields[7],
            height=fields[8],
            opencv_dim=opencv_dim,
            clip_dim=clip_dim,
        )
    
    def _parse_header_v1(self):
        raw = self._fh.read(HEADER_SIZE_V1)
        fields = struct.unpack(HEADER_FMT_V1, raw)
        self.header = AVISHeaderV2(
            version=1,
            model_name=fields[2].decode("utf-8").rstrip("\x00"),
            is_hybrid=False,
            keyframe_interval=fields[4],
            fps=fields[5],
            total_frames=fields[6],
            width=fields[7],
            height=fields[8],
        )
    
    def _scan_index_v2(self):
        """Scan V2 file with multi-layer frames. Stops after total_frames entries."""
        self._fh.seek(HEADER_SIZE_V2)
        self._index: dict[int, FrameEntryV2] = {}
        
        while len(self._index) < self.header.total_frames:
            pos = self._fh.tell()
            prefix = self._fh.read(FRAME_PREFIX_SIZE)
            if len(prefix) < FRAME_PREFIX_SIZE:
                break
            
            frame_idx, frame_type, ts = struct.unpack(FRAME_PREFIX_FMT, prefix)
            
            if frame_type == FRAME_TYPE_IFRAME:
                num_layers = struct.unpack(IFRAME_EXTRA_FMT, self._fh.read(IFRAME_EXTRA_SIZE))[0]
                layers = []
                for _ in range(num_layers):
                    layer_start = self._fh.tell()
                    dim, comp_len = struct.unpack(LAYER_HDR_FMT, self._fh.read(LAYER_HDR_SIZE))
                    self._fh.seek(comp_len, 1)  # skip compressed data
                    layers.append((dim, comp_len, layer_start))
                
                self._index[frame_idx] = FrameEntryV2(
                    frame_idx=frame_idx, timestamp=ts, is_iframe=True,
                    file_offset=pos, magnitude=0.0, num_layers=num_layers,
                    layers=layers,
                )
            elif frame_type == FRAME_TYPE_DELTA:
                mag, num_layers = struct.unpack(DELTA_EXTRA_FMT, self._fh.read(DELTA_EXTRA_SIZE))
                layers = []
                for _ in range(num_layers):
                    layer_start = self._fh.tell()
                    dim, comp_len = struct.unpack(LAYER_HDR_FMT, self._fh.read(LAYER_HDR_SIZE))
                    self._fh.seek(comp_len, 1)
                    layers.append((dim, comp_len, layer_start))
                
                self._index[frame_idx] = FrameEntryV2(
                    frame_idx=frame_idx, timestamp=ts, is_iframe=False,
                    file_offset=pos, magnitude=mag, num_layers=num_layers,
                    layers=layers,
                )
            else:
                raise ValueError(f"Unknown frame type: {frame_type:#x} at offset {pos}")
        
        self._sorted_frames = sorted(self._index.keys())
    
    def _scan_index_v1(self):
        """Scan V1 file (legacy format)."""
        from .decoder import AVISReader
        # Use existing V1 reader for backward compat
        v1 = AVISReader(str(self.path))
        self.header.total_frames = v1.header.total_frames
        self._sorted_frames = v1.frame_indices
        # Build V2-style entries from V1 data
        self._index = {}
        for fidx in self._sorted_frames:
            entry = v1._index[fidx]
            self._index[fidx] = FrameEntryV2(
                frame_idx=fidx, timestamp=entry.timestamp,
                is_iframe=entry.is_iframe,
                file_offset=entry.file_offset,
                magnitude=entry.magnitude,
                num_layers=1,
                layers=[(v1.header.latent_dim, entry.compressed_size, entry.file_offset)],
            )
        v1.close()
    
    def get_feature(self, frame_idx: int, layer: int = 0) -> np.ndarray:
        """
        Get feature vector for a frame.
        layer=0: OpenCV/base features (always available)
        layer=1: CLIP features (only on I-frames; returns nearest for Δ-frames)
        """
        entry = self._index.get(frame_idx)
        if entry is None:
            raise KeyError(f"Frame {frame_idx} not found")
        
        if layer >= entry.num_layers and not entry.is_iframe and layer == 1:
            # CLIP layer requested on Δ-frame → interpolate from nearest I-frame
            iframe_idx = self._find_preceding_iframe(frame_idx)
            return self._read_layer(iframe_idx, layer=1)
        
        if layer >= entry.num_layers:
            raise KeyError(f"Layer {layer} not available for frame {frame_idx}")
        
        if entry.is_iframe:
            return self._read_layer(frame_idx, layer=layer)
        else:
            # Δ-frame: base layer is I-frame base + delta
            if layer == 0:
                iframe_idx = self._find_preceding_iframe(frame_idx)
                base = self._read_layer(iframe_idx, layer=0)
                delta = self._read_delta_layer(frame_idx, layer=0)
                return base + delta
            else:
                # CLIP on Δ-frame → return nearest I-frame's CLIP
                iframe_idx = self._find_preceding_iframe(frame_idx)
                return self._read_layer(iframe_idx, layer=layer)
    
    def _find_preceding_iframe(self, frame_idx: int) -> int:
        for fidx in reversed(self._sorted_frames):
            if fidx <= frame_idx and self._index[fidx].is_iframe:
                return fidx
        raise ValueError(f"No preceding I-frame for {frame_idx}")
    
    def _read_layer(self, frame_idx: int, layer: int) -> np.ndarray:
        """Read a stored layer from an I-frame."""
        entry = self._index[frame_idx]
        dim, comp_len, data_offset = entry.layers[layer]
        
        # data_offset includes the layer header; skip it
        self._fh.seek(data_offset + LAYER_HDR_SIZE)
        compressed = self._fh.read(comp_len)
        raw = zlib.decompress(compressed)
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    
    def _read_delta_layer(self, frame_idx: int, layer: int) -> np.ndarray:
        """Read a delta layer from a Δ-frame."""
        entry = self._index[frame_idx]
        dim, comp_len, data_offset = entry.layers[layer]
        self._fh.seek(data_offset + LAYER_HDR_SIZE)
        compressed = self._fh.read(comp_len)
        raw = zlib.decompress(compressed)
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    
    @property
    def frame_count(self) -> int:
        return self.header.total_frames
    
    @property
    def frame_indices(self) -> list[int]:
        return self._sorted_frames
    
    def close(self):
        self._fh.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class AVISDecoderV2:
    """High-level V2 decoder."""
    
    def __init__(self, path: str):
        self.reader = AVISReaderV2(path)
        self.header = self.reader.header
    
    def opencv_features(self, start=0, end=None) -> list[np.ndarray]:
        """Get base-layer (OpenCV) features for a range."""
        if end is None:
            end = self.header.total_frames
        return [self.reader.get_feature(i, layer=0) for i in range(start, min(end, self.header.total_frames))]
    
    def clip_features(self, start=0, end=None) -> list[np.ndarray]:
        """Get CLIP features (interpolated for Δ-frames)."""
        if end is None:
            end = self.header.total_frames
        return [self.reader.get_feature(i, layer=1) for i in range(start, min(end, self.header.total_frames))]
    
    def scene_changes(self, threshold=0.06) -> list[int]:
        """Detect scene changes using base-layer features."""
        feats = self.opencv_features()
        changes = []
        for i in range(1, len(feats)):
            if (1.0 - float(np.dot(feats[i], feats[i-1]))) > threshold:
                changes.append(i)
        return changes
    
    @property
    def i_frame_indices(self) -> list[int]:
        return [i for i in self.reader.frame_indices if self.reader._index[i].is_iframe]
    
    def close(self):
        self.reader.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
