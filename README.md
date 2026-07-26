# AVIS — AI Video Interchange Stream

**A video format designed for AI, not humans.**

Traditional video formats (MP4, HEVC) are designed for human eyes — pixel-perfect reconstruction. AVIS is designed for AI consumption — storing **semantic features** instead of pixels. The result: AI processes AVIS video **5-17× cheaper** than traditional formats.

```
MP4 pipeline:  decode → pixels → VAE/CLIP → tokens     (every frame)
AVIS pipeline: read features directly → tokens           (I-frames only)
```

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                  AVIS File                           │
│                                                      │
│  I-Frame (every N frames):                          │
│    Full CLIP features (512-dim semantic embedding)   │
│    + OpenCV features for scheduling                  │
│                                                      │
│  Δ-Frame (all other frames):                        │
│    Lightweight delta from last I-frame               │
│    Only OpenCV delta (tiny, ~150 bytes)              │
│                                                      │
│  Transcript (V3):                                    │
│    Time-aligned speech-to-text (Whisper)              │
└─────────────────────────────────────────────────────┘
```

**Text search on video?** CLIP encodes both text and images into the same 512-dim space. When you search for "comedian on stage with microphone", the query text is encoded to a 512d vector and compared against every I-frame's stored CLIP embedding via cosine similarity. No video decoding needed.

## Backends

| Backend | Dim | Speed | Use Case |
|---------|-----|-------|----------|
| OpenCV (HSV+Edge) | 74d | 100 fps | Real-time, simple scene detection |
| PyTorch (MobileNetV3) | 576d | 15 fps | General-purpose, balanced |
| CLIP (ViT-B-32) | 512d | 12 fps | Semantic search, cross-modal |
| **Hybrid** (OpenCV + CLIP) | 74d+512d | 60 fps | Best of both: fast scheduling + semantic I-frames |
| **Multimodal** (+Whisper) | +transcript | — | Full video content indexing |

## Quick Start

```bash
# Encode video with hybrid encoder (fast + semantic)
python3 -c "
from avis import encode_video_torch
encode_video_torch('my_video.mp4', 'my_video.avis')
"

# Read and search
from avis.decoder_v2 import AVISDecoderV2
from avis.query_hybrid import AVISQueryHybrid

with AVISQueryHybrid('my_video.avis', clip_model_path='clip_model.bin') as q:
    # CLIP text search — finds frames matching description
    results = q.search_by_text('a person on stage')
    print(f'Best match: frame {results[0][0]}')
```

## Real-World Benchmarks

| Video | Frames | Hybrid Encode | Token Reduction |
|-------|--------|---------------|-----------------|
| Synthetic (8s) | 240 | 2s | 7.5× |
| Stand-up clip (25s) | 635 | 10s | 17.7× |
| Full stand-up (5min) | 7,902 | 120s | 15×+ |

## File Structure

```
avis/
├── format_v2.py          # V2 multi-layer binary format
├── format_v3.py          # V3 multimodal format (+transcript)
├── extractors.py         # OpenCV color/edge features
├── extractors_torch.py   # MobileNetV3 features
├── encoder_lite.py       # OpenCV backend
├── encoder_torch.py      # PyTorch backend
├── encoder_clip.py       # CLIP backend (needs model download)
├── encoder_hybrid.py     # Hybrid: OpenCV scheduler + CLIP on I-frames
├── encoder_multimodal.py # Multimodal: +Whisper transcript
├── decoder.py            # V1 format reader
├── decoder_v2.py         # V2/V3 format reader
├── query.py              # V1 query engine
├── query_hybrid.py       # Hybrid query engine
└── reader_multimodal.py  # Multimodal reader
scripts/
├── demo.py               # 3-way backend comparison (synthetic)
├── standup_demo.py       # 3-way comparison (real video)
├── hybrid_demo.py        # Hybrid vs pure comparison
├── v3_demo.py            # V3 multimodal demo
└── understand.py         # Full video analysis
```

## Limitations (Honest)

- **Not "true understanding"** — AVIS tells you *what* is in the frame, not *why* it matters
- **CLIP text search** works on I-frames only (Δ-frames interpolated from nearest I-frame)
- **Out-of-distribution content** (anime, heavy text overlays) degrades CLIP quality
- **CPU-only** — currently no GPU acceleration implemented
- **No pixel reconstruction** — AVIS is for AI reading, not human viewing

## Why This Matters

Processing video with AI is expensive. 1 hour of 1080p video → ~2-5M tokens → $5-15 in API costs. AVIS reduces that by 5-17× by storing features, not pixels. That's hundreds of dollars saved per video archive per processing pass.

This is not "AI understanding video." This is infrastructure — like JPEG for images or MP4 for playback. AVIS is the compression layer for the AI video pipeline.

---

*AVIS does not achieve "true video understanding." It doesn't need to. A map doesn't need to be the territory to be useful.*
