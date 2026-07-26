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

## AVIS as a Standard: The Strategic Case

AVIS solves a rapidly growing cost problem: **eliminating redundant computation**.

Every major AI model processing video today repeats the same work — decode pixels into tensors, run them through an encoder. A 10-minute video processed by GPT-4V, then Gemini, then Claude means the same compute is burned three times. Whoever builds the "encode once, reuse everywhere" intermediate layer becomes the unavoidable link in the pipeline.

Historical standards were born at exactly this inflection point — not inventing new technology, but eliminating redundant waste:

- **JSON** replaced every program writing its own config parser
- **Markdown** replaced every forum inventing its own formatting syntax
- **MP3** wasn't the best compression algorithm, but Fraunhofer drove hardware licensing
- **TCP/IP** beat OSI not on design quality, but because it shipped first

### Winner-Takes-All, But the Inventor Isn't Always the Winner

| Condition | Current Status |
|---|---|
| Runnable demo | ✅ avis.py CLI + avis binary format |
| Benchmark data (cost vs raw video) | Needs to be established |
| At least one AI video company/project using it | Start with live-clip itself |
| Published paper or technical blog | Needs to be written |
| Community discussion / GitHub stars | Needs promotion |

### Path to Adoption: Build a Tool, Let the Standard Grow Organically

Don't write a dead standards document. Build a living developer tool — "the intermediate format CLI for AI video processing." Any AI video developer should be able to use it. Once you have users, the format standard isn't something you push — it's something they demand.

AVIS and live-clip are synergistic — use your own project to store features in AVIS format, record a processing-speed comparison video, and that's your best promotional material.

**Bottom line: AVIS has high long-term potential, but to land it, treat it as a living product, not a dead standard. What you're missing isn't the idea — it's the first user.**

---

*AVIS does not achieve "true video understanding." It doesn't need to. A map doesn't need to be the territory to be useful.*

