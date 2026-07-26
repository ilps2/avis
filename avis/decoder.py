"""
AVIS Decoder — reads Feature-Stream format and reconstructs frame features.

Supports:
  - Reading full feature for any frame (I-frame direct + delta reconstruction)
  - Streaming frame-by-frame iteration
  - Frame index for O(1) random access
"""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


MAGIC = b"AVIS"
HEADER_FMT = ">4s I 64s I I f I I I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

FRAME_TYPE_IFRAME = 0x00
FRAME_TYPE_DELTA  = 0x01

IFRAME_FMT = ">I B d"          # frame_idx, type, timestamp
IFRAME_HDR_SIZE = struct.calcsize(IFRAME_FMT)

DFRAME_FMT = ">I B d f"        # frame_idx, type, timestamp, magnitude
DFRAME_HDR_SIZE = struct.calcsize(DFRAME_FMT)

FEATURE_DTYPE = np.float16


@dataclass
class AVISHeader:
    """Parsed AVIS file header."""
    version: int
    model_name: str
    latent_dim: int
    keyframe_interval: int
    fps: float
    total_frames: int
    width: int
    height: int


@dataclass
class FrameEntry:
    """Descriptor for one frame in the file."""
    frame_idx: int
    timestamp: float
    is_iframe: bool
    file_offset: int
    magnitude: float
    compressed_size: int


class AVISReader:
    """Low-level reader for AVIS binary format with O(1) random access."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._fh = open(path, "rb")
        
        # Parse header
        raw = self._fh.read(HEADER_SIZE)
        fields = struct.unpack(HEADER_FMT, raw)
        
        self.header = AVISHeader(
            version=fields[1],
            model_name=fields[2].decode("utf-8").rstrip("\x00"),
            latent_dim=fields[3],
            keyframe_interval=fields[4],
            fps=fields[5],
            total_frames=fields[6],
            width=fields[7],
            height=fields[8],
        )
        
        # Build index
        self._index: dict[int, FrameEntry] = {}
        self._sorted_frames: list[int] = []
        self._scan_index()
        
    def _scan_index(self):
        """Scan file and build frame index."""
        self._fh.seek(HEADER_SIZE)
        
        while True:
            pos = self._fh.tell()
            header_start = self._fh.read(4)
            if len(header_start) < 4:
                break
            
            frame_idx = struct.unpack(">I", header_start)[0]
            type_byte = self._fh.read(1)
            frame_type = struct.unpack(">B", type_byte)[0]
            
            if frame_type == FRAME_TYPE_IFRAME:
                ts_bytes = self._fh.read(8)
                ts = struct.unpack(">d", ts_bytes)[0]
                comp_len_bytes = self._fh.read(4)
                comp_len = struct.unpack(">I", comp_len_bytes)[0]
                self._fh.seek(comp_len, 1)
                
                self._index[frame_idx] = FrameEntry(
                    frame_idx=frame_idx,
                    timestamp=ts,
                    is_iframe=True,
                    file_offset=pos,
                    magnitude=0.0,
                    compressed_size=comp_len,
                )
                
            elif frame_type == FRAME_TYPE_DELTA:
                ts_bytes = self._fh.read(8)
                ts = struct.unpack(">d", ts_bytes)[0]
                mag_bytes = self._fh.read(4)
                mag = struct.unpack(">f", mag_bytes)[0]
                comp_len_bytes = self._fh.read(4)
                comp_len = struct.unpack(">I", comp_len_bytes)[0]
                self._fh.seek(comp_len, 1)
                
                self._index[frame_idx] = FrameEntry(
                    frame_idx=frame_idx,
                    timestamp=ts,
                    is_iframe=False,
                    file_offset=pos,
                    magnitude=mag,
                    compressed_size=comp_len,
                )
            else:
                raise ValueError(f"Unknown frame type byte: {frame_type:#x} at offset {pos}")
        
        self._sorted_frames = sorted(self._index.keys())
        
    def get_feature(self, frame_idx: int) -> np.ndarray:
        """
        Get reconstructed feature vector for a given frame.
        I-frames: return stored features directly.
        Δ-frames: find nearest preceding I-frame, read it, add delta.
        """
        entry = self._index.get(frame_idx)
        if entry is None:
            raise KeyError(f"Frame {frame_idx} not found (have {len(self._index)} frames)")
        
        if entry.is_iframe:
            return self._read_iframe_features(frame_idx)
        else:
            iframe_idx = self._find_preceding_iframe(frame_idx)
            iframe_features = self._read_iframe_features(iframe_idx)
            delta = self._read_delta(frame_idx)
            return iframe_features + delta
    
    def _find_preceding_iframe(self, frame_idx: int) -> int:
        """Find the nearest I-frame at or before frame_idx."""
        for fidx in reversed(self._sorted_frames):
            if fidx <= frame_idx and self._index[fidx].is_iframe:
                return fidx
        raise ValueError(f"No preceding I-frame found for frame {frame_idx}")
    
    def _read_iframe_features(self, frame_idx: int) -> np.ndarray:
        entry = self._index[frame_idx]
        self._fh.seek(entry.file_offset + IFRAME_HDR_SIZE)
        comp_len = struct.unpack(">I", self._fh.read(4))[0]
        compressed = self._fh.read(comp_len)
        raw = zlib.decompress(compressed)
        return np.frombuffer(raw, dtype=FEATURE_DTYPE).astype(np.float32)
    
    def _read_delta(self, frame_idx: int) -> np.ndarray:
        entry = self._index[frame_idx]
        self._fh.seek(entry.file_offset + DFRAME_HDR_SIZE)
        comp_len = struct.unpack(">I", self._fh.read(4))[0]
        compressed = self._fh.read(comp_len)
        raw = zlib.decompress(compressed)
        return np.frombuffer(raw, dtype=FEATURE_DTYPE).astype(np.float32)
    
    @property
    def frame_count(self) -> int:
        return self.header.total_frames
    
    @property 
    def frame_indices(self) -> list[int]:
        return self._sorted_frames
    
    @property
    def i_frame_count(self) -> int:
        return sum(1 for e in self._index.values() if e.is_iframe)
    
    @property
    def delta_frame_count(self) -> int:
        return sum(1 for e in self._index.values() if not e.is_iframe)
    
    def close(self):
        self._fh.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class AVISDecoder:
    """High-level decoder with convenience methods for analysis."""
    
    def __init__(self, path: str):
        self.reader = AVISReader(path)
        self.header = self.reader.header
    
    def features_for_range(self, start: int, end: int) -> list[np.ndarray]:
        return [self.reader.get_feature(i) for i in range(start, min(end, self.header.total_frames))]
    
    def all_features(self) -> list[np.ndarray]:
        return self.features_for_range(0, self.header.total_frames)
    
    def scene_changes(self, threshold: float = 0.15) -> list[int]:
        """Detect scene changes via cosine distance between consecutive frames."""
        features = self.all_features()
        changes = []
        for i in range(1, len(features)):
            cos_sim = np.dot(features[i], features[i-1])
            if (1.0 - cos_sim) > threshold:
                changes.append(i)
        return changes
    
    def similarity_matrix(self, max_frames: int = 100) -> np.ndarray:
        """Pairwise cosine similarity for first max_frames frames."""
        features = self.features_for_range(0, min(max_frames, self.header.total_frames))
        n = len(features)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = float(np.dot(features[i], features[j]))
        return matrix
    
    def close(self):
        self.reader.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
