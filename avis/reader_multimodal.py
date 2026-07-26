"""
AVIS Multimodal Reader — visual features + transcript in one file.
"""

from pathlib import Path
from typing import Optional
import struct

import numpy as np

from .decoder_v2 import AVISDecoderV2
from .format_v3 import read_transcript_block, Transcript, TranscriptSegment
from .query_hybrid import AVISQueryHybrid


class AVISMultimodalReader:
    """
    Reads AVIS V3 files with transcript layer.
    Falls back gracefully: if no transcript, visual-only still works.
    """
    
    def __init__(self, path: str, clip_model_path: str = None):
        self.path = Path(path)
        self.visual = AVISDecoderV2(path)
        self._transcript: Optional[Transcript] = None
        self._clip_model_path = clip_model_path
        
        # Try to load transcript from end of file
        self._load_transcript()
    
    def _load_transcript(self):
        """Seek to end of file, find TEXT marker, read transcript."""
        with open(self.path, "rb") as fh:
            fsize = fh.seek(0, 2)
            if fsize < 16:
                return
            
            # Scan last 64KB for TEXT marker
            search_start = max(0, fsize - 65536)
            fh.seek(search_start)
            data = fh.read()
            
            # Find "TEXT" marker
            idx = data.find(b"TEXT")
            if idx < 0:
                return
            
            # After TEXT: json_len (4B big-endian), then json_data, then AVIS footer
            pos = idx + 4
            if pos + 4 > len(data):
                return
            json_len = struct.unpack(">I", data[pos:pos+4])[0]
            pos += 4
            
            if pos + json_len + 4 > len(data):
                return
            
            json_data = data[pos:pos+json_len]
            pos += json_len
            
            footer = data[pos:pos+4]
            if footer != b"AVIS":
                return
            
            self._transcript = Transcript.from_json(json_data)
    
    @property
    def has_transcript(self) -> bool:
        return self._transcript is not None
    
    @property
    def transcript(self) -> Optional[Transcript]:
        return self._transcript
    
    def search_text(self, query: str) -> list[TranscriptSegment]:
        """Search transcript for keyword. Returns matching segments with timestamps."""
        if not self._transcript:
            return []
        return self._transcript.search(query)
    
    def what_is_happening_at(self, t_seconds: float) -> dict:
        """What's happening at a given time? Combines visual + text context."""
        result = {"time": t_seconds}
        
        # Visual: what does the frame look like?
        frame_idx = int(t_seconds * self.visual.header.fps)
        frame_idx = min(frame_idx, self.visual.header.total_frames - 1)
        
        ocv_feat = self.visual.reader.get_feature(frame_idx, layer=0)
        clip_feat = self.visual.reader.get_feature(frame_idx, layer=1)
        result["frame"] = frame_idx
        
        # Transcript: what's being said?
        if self._transcript:
            seg = self._transcript.at_time(t_seconds)
            if seg:
                result["speaking"] = seg.text
        
        return result
    
    def search_all(self, text_query: str, clip_model_path: str = None, top_k: int = 5) -> dict:
        """
        Combined search: transcript keyword + visual similarity.
        Returns segments that match text AND their visual context.
        """
        result = {"text_matches": [], "visual_matches": []}
        
        # Text search in transcript
        if self._transcript:
            result["text_matches"] = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in self._transcript.search(text_query)
            ]
        
        # Visual search (requires CLIP model)
        if clip_model_path or self._clip_model_path:
            cp = clip_model_path or self._clip_model_path
            try:
                q = AVISQueryHybrid(str(self.path), clip_model_path=cp)
                visual = q.search_by_text(text_query, top_k=top_k)
                fps = self.visual.header.fps
                result["visual_matches"] = [
                    {"frame": fidx, "time": fidx / fps, "similarity": sim}
                    for fidx, sim in visual
                ]
                q.close()
            except Exception:
                pass
        
        return result
    
    def close(self):
        self.visual.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
