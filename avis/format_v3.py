"""
AVIS V3 Multimodal Format

Adds time-aligned transcript to V2 hybrid files.
Transcript stored at end of file as JSON block with magic markers.

File layout:
  [Header V2 (version=3)]  ← same as V2, version bumped
  [Frame data...]           ← V2 multi-layer frames, unchanged
  [Transcript Block]        ← new: time-aligned text
    magic:     "TEXT" (4B)
    json_len:  uint32 (4B)
    json_data: UTF-8 JSON
  [File Footer]
    magic:     "AVIS" (4B)

Backward compatible: V2 reader ignores unknown data after last frame.
V3 reader detects "TEXT" marker after frames, loads transcript.
"""

import struct
import json
from dataclasses import dataclass
from typing import Optional


TRANSCRIPT_MAGIC = b"TEXT"
FOOTER_MAGIC = b"AVIS"
VERSION_V3 = 3


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    model: str
    language: str
    segments: list[TranscriptSegment]
    
    @classmethod
    def from_whisper(cls, whisper_result: dict, model_name: str = "whisper-base") -> "Transcript":
        return cls(
            model=model_name,
            language=whisper_result.get("language", "en"),
            segments=[
                TranscriptSegment(start=s["start"], end=s["end"], text=s["text"].strip())
                for s in whisper_result["segments"]
            ]
        )
    
    def to_json(self) -> bytes:
        data = {
            "model": self.model,
            "language": self.language,
            "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in self.segments],
        }
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    
    @classmethod
    def from_json(cls, data: bytes) -> "Transcript":
        obj = json.loads(data.decode("utf-8"))
        return cls(
            model=obj["model"],
            language=obj["language"],
            segments=[TranscriptSegment(**s) for s in obj["segments"]],
        )
    
    def search(self, query: str) -> list[TranscriptSegment]:
        """Simple keyword search across transcript segments."""
        q = query.lower()
        return [s for s in self.segments if q in s.text.lower()]
    
    def at_time(self, t: float) -> Optional[TranscriptSegment]:
        """Get transcript segment at a given time."""
        for s in self.segments:
            if s.start <= t <= s.end:
                return s
        return None
    
    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


def write_transcript_block(fh, transcript: Transcript):
    """Write transcript block + footer to end of file."""
    json_data = transcript.to_json()
    fh.write(TRANSCRIPT_MAGIC)
    fh.write(struct.pack(">I", len(json_data)))
    fh.write(json_data)
    fh.write(FOOTER_MAGIC)


def read_transcript_block(fh) -> Optional[Transcript]:
    """Try to read transcript block from current file position. Returns None if not found."""
    pos = fh.tell()
    
    # Read potential magic
    magic = fh.read(4)
    if magic != TRANSCRIPT_MAGIC:
        # No transcript block
        fh.seek(pos)
        return None
    
    # Read JSON length and data
    json_len_bytes = fh.read(4)
    if len(json_len_bytes) < 4:
        fh.seek(pos)
        return None
    
    json_len = struct.unpack(">I", json_len_bytes)[0]
    json_data = fh.read(json_len)
    
    if len(json_data) < json_len:
        fh.seek(pos)
        return None
    
    # Verify footer
    footer = fh.read(4)
    if footer != FOOTER_MAGIC:
        fh.seek(pos)
        return None
    
    return Transcript.from_json(json_data)
