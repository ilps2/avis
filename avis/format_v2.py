"""
AVIS V2 Binary Format — Multi-Layer Feature Support

V2 extends V1 with per-frame layer metadata, enabling hybrid encoders
that store different feature types (OpenCV + CLIP) in the same stream.

Frame entry (V2):
  frame_idx:   uint32  (4B)
  frame_type:  uint8   (1B)  — IFRAME=0x00, DELTA=0x01
  timestamp:   float64 (8B)
  
  If DELTA:
    magnitude:  float32 (4B)  — L2 norm of base-layer delta
    num_layers: uint8   (1B)  — always 1 (base layer only)
    [layer 0]:
      dim:      uint16  (2B)
      comp_len: uint32  (4B)
      data:     zlib(float16[dim])
  
  If IFRAME:
    num_layers: uint8   (1B)  — ≥1
    [layer 0..N-1]:
      dim:      uint16  (2B)
      comp_len: uint32  (4B)
      data:     zlib(float16[dim])

Header (same as V1):
  magic:      4s      "AVIS"
  version:    uint32  2
  model_name: 64s     e.g. "hybrid:opencv74+clip512"
  _reserved:  uint32  (was latent_dim in V1, now reserved)
  interval:   uint32
  fps:        float32
  frames:     uint32
  width:      uint32
  height:     uint32
"""

import struct

MAGIC = b"AVIS"

# ── Shared constants ────────────────────────────────────────────
FRAME_TYPE_IFRAME = 0x00
FRAME_TYPE_DELTA  = 0x01

# V1 formats (backward compat)
HEADER_FMT_V1  = ">4s I 64s I I f I I I"
HEADER_SIZE_V1 = struct.calcsize(HEADER_FMT_V1)

# V2 formats
HEADER_FMT_V2  = ">4s I 64s I I f I I I"  # same layout, latent_dim → reserved
HEADER_SIZE_V2 = struct.calcsize(HEADER_FMT_V2)

# V2 frame entry prefixes (before layer data)
FRAME_PREFIX_FMT = ">I B d"     # frame_idx, type, timestamp
FRAME_PREFIX_SIZE = struct.calcsize(FRAME_PREFIX_FMT)

DELTA_EXTRA_FMT  = ">f B"       # magnitude, num_layers
DELTA_EXTRA_SIZE  = struct.calcsize(DELTA_EXTRA_FMT)

IFRAME_EXTRA_FMT = ">B"         # num_layers
IFRAME_EXTRA_SIZE = struct.calcsize(IFRAME_EXTRA_FMT)

LAYER_HDR_FMT = ">H I"          # dim, comp_len
LAYER_HDR_SIZE = struct.calcsize(LAYER_HDR_FMT)

FEATURE_DTYPE_BYTES = 2  # float16


def pack_layer(dim: int, data_compressed: bytes) -> bytes:
    """Pack a single feature layer."""
    return struct.pack(LAYER_HDR_FMT, dim, len(data_compressed)) + data_compressed


def unpack_layer(data: bytes, offset: int) -> tuple:
    """Unpack a single feature layer header. Returns (dim, comp_len, new_offset)."""
    dim, comp_len = struct.unpack_from(LAYER_HDR_FMT, data, offset)
    return dim, comp_len, offset + LAYER_HDR_SIZE
