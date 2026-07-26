"""
AVIS Hybrid Query — dual-channel semantic search.

Fast channel (OpenCV): frame similarity, scene detection — always available
Slow channel (CLIP):   text-to-frame search — available on I-frames, interpolated for Δ-frames
"""

import numpy as np
import torch
import open_clip

from .decoder_v2 import AVISDecoderV2


class AVISQueryHybrid:
    """Query engine for hybrid V2 files with dual feature layers."""
    
    def __init__(self, avis_path: str, clip_model_path: str = None, device: str = "cpu"):
        self.decoder = AVISDecoderV2(avis_path)
        self.header = self.decoder.header
        self.device = device
        self._clip_model_path = clip_model_path
        
        # Lazy CLIP text encoder
        self._text_model = None
        self._tokenizer = None
    
    def _ensure_text_encoder(self):
        if self._text_model is not None:
            return
        if self._clip_model_path is None:
            raise ValueError("clip_model_path required for text search")
        
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=self._clip_model_path
        )
        self._text_model = model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    def find_similar_frames(self, query_frame: int, top_k: int = 5, use_clip: bool = False):
        """
        Find frames similar to query_frame.
        use_clip=False: fast OpenCV-based search (always works)
        use_clip=True:  semantic CLIP-based search (I-frames only, interpolated)
        """
        if use_clip:
            query_feat = self.decoder.reader.get_feature(query_frame, layer=1)
            all_feats = [self.decoder.reader.get_feature(i, layer=1) 
                        for i in self.decoder.reader.frame_indices]
        else:
            query_feat = self.decoder.reader.get_feature(query_frame, layer=0)
            all_feats = [self.decoder.reader.get_feature(i, layer=0)
                        for i in self.decoder.reader.frame_indices]
        
        scores = []
        for fidx, feat in zip(self.decoder.reader.frame_indices, all_feats):
            if fidx == query_frame:
                continue
            scores.append((fidx, float(np.dot(query_feat, feat))))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def search_by_text(self, text: str, top_k: int = 5):
        """
        CLIP text-to-frame search.
        Only searches I-frames (where CLIP features are stored).
        """
        self._ensure_text_encoder()
        
        text_tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_feat = self._text_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.cpu().numpy().squeeze(0)
        
        # Search all frames (CLIP features exist on I-frames, interpolated on Δ)
        scores = []
        for fidx in self.decoder.reader.frame_indices:
            clip_feat = self.decoder.reader.get_feature(fidx, layer=1)
            scores.append((fidx, float(np.dot(text_feat, clip_feat))))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def temporal_summary(self, num_segments: int = 5):
        """Activity summary using fast OpenCV features."""
        total = self.header.total_frames
        seg_size = max(1, total // num_segments)
        all_feats = self.decoder.opencv_features()
        
        segments = []
        for i in range(num_segments):
            start = i * seg_size
            end = min(start + seg_size, total)
            seg = np.stack(all_feats[start:end])
            var = ((seg - seg.mean(axis=0)) ** 2).mean()
            segments.append({
                "start": start, "end": end,
                "duration": (end - start) / max(self.header.fps, 1),
                "variance": float(var),
            })
        return segments
    
    def close(self):
        self.decoder.close()
        self._text_model = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
