"""
AVIS — AI Video Interchange Stream

Three encoder backends, one format:
  - encoder_clip.py:  CLIP ViT-B-32 (512-dim, semantic)
  - encoder_torch.py: MobileNetV3-Small (576-dim, learned)
  - encoder_lite.py:  OpenCV HSV+Edge (74-dim, instant)
"""

from .decoder import AVISDecoder, AVISReader
from .query import AVISQuery
from .encoder_lite import encode_video_lite
from .encoder_torch import encode_video_torch
from .encoder_clip import encode_video_clip, CLIPExtractor
