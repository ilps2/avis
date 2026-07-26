"""
AVIS Query — semantic operations over Feature-Stream video.

Supports:
  - Frame similarity search
  - CLIP text-to-frame search (with local model path)
  - Scene change detection
  - Temporal activity summary
"""

from typing import Optional
import numpy as np
import torch
import open_clip

from .decoder import AVISDecoder


class AVISQuery:
    """Semantic query engine over AVIS files."""
    
    def __init__(self, avis_path: str, device: str = "cpu"):
        self.decoder = AVISDecoder(avis_path)
        self.header = self.decoder.header
        self.device = device
        
        # Lazily loaded text encoder
        self._text_model = None
        self._tokenizer = None
        self._clip_model_path = None
    
    def _ensure_text_encoder(self, clip_model_path: str):
        """Lazy-load CLIP text encoder for text search."""
        if self._text_model is not None and self._clip_model_path == clip_model_path:
            return
        
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=clip_model_path
        )
        model = model.to(self.device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        
        self._text_model = model
        self._tokenizer = tokenizer
        self._clip_model_path = clip_model_path
    
    def find_similar_frames(
        self, query_frame: int, top_k: int = 5, exclude_self: bool = True,
    ) -> list[tuple[int, float]]:
        """Find frames most visually similar to a given frame."""
        query_feat = self.decoder.reader.get_feature(query_frame)
        all_feats = self.decoder.all_features()
        
        scores = []
        for i, feat in enumerate(all_feats):
            if exclude_self and i == query_frame:
                continue
            sim = float(np.dot(query_feat, feat))
            scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def search_by_text(
        self, text: str, top_k: int = 5, clip_model_path: str = None,
    ) -> list[tuple[int, float]]:
        """
        Find frames that best match a text description using CLIP.
        
        Args:
            text: Natural language query
            top_k: Number of results
            clip_model_path: Path to open_clip_pytorch_model.bin (required)
            
        Returns:
            list of (frame_idx, cosine_similarity)
        """
        if clip_model_path is None:
            raise ValueError("clip_model_path is required for text search")
        
        self._ensure_text_encoder(clip_model_path)
        
        text_tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_feat = self._text_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu().numpy().squeeze(0)
        
        all_feats = self.decoder.all_features()
        scores = [(i, float(np.dot(text_feat, feat))) for i, feat in enumerate(all_feats)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def temporal_summary(self, num_segments: int = 5) -> list[dict]:
        """Divide video into temporal segments and summarize each."""
        total = self.header.total_frames
        segment_size = max(1, total // num_segments)
        all_feats = self.decoder.all_features()
        
        segments = []
        for i in range(num_segments):
            start = i * segment_size
            end = min(start + segment_size, total)
            segment_feats = np.stack(all_feats[start:end])
            mean_feat = segment_feats.mean(axis=0)
            variances = ((segment_feats - mean_feat) ** 2).mean()
            
            segments.append({
                "start_frame": start,
                "end_frame": end,
                "duration_sec": (end - start) / max(self.header.fps, 1),
                "feature_variance": float(variances),
                "activity_level": (
                    "high" if variances > 0.01 else
                    "medium" if variances > 0.002 else
                    "low"
                ),
            })
        return segments
    
    def compare_frames(self, frame_a: int, frame_b: int) -> dict:
        """Detailed comparison between two frames."""
        feat_a = self.decoder.reader.get_feature(frame_a)
        feat_b = self.decoder.reader.get_feature(frame_b)
        
        cos_sim = float(np.dot(feat_a, feat_b))
        l2_dist = float(np.linalg.norm(feat_a - feat_b))
        
        return {
            "frame_a": frame_a,
            "frame_b": frame_b,
            "cosine_similarity": cos_sim,
            "l2_distance": l2_dist,
            "same_scene_likely": cos_sim > 0.85,
        }
    
    def close(self):
        self.decoder.close()
        self._text_model = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
