#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_layer.py — 对象级视频语义层原型 (v0.1)

一次本地分析 -> 对象向量标注 + 场景描述 + ASR 时间轴
之后无限次本地查询，零 token。

用法:
  python3 semantic_layer.py build <video.mp4> -o <idx_dir> [--fps 1] [--asr large-v3] [--max-frames 400]
  python3 semantic_layer.py query <idx_dir> --text "红色杯子" [-n 5]
  python3 semantic_layer.py query <idx_dir> --image ref.png [-n 5]
  python3 semantic_layer.py info <idx_dir>
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

HF_HOME = Path.home() / ".cache/huggingface/hub"
MODELS_ROOTS = [
    Path(__file__).resolve().parent.parent / "models",
    Path("/Users/chuli/Documents/Codex/2026-08-06/wo/work/models"),
]
DEFAULT_PROMPTS = [
    "person", "face", "hand", "phone", "smartphone", "bottle", "cup", "drink",
    "product", "package", "box", "bag", "lipstick", "cosmetics", "makeup",
    "watch", "glasses", "laptop", "tablet", "microphone", "book", "paper",
    "card", "remote", "keyboard", "mouse", "chair", "table", "screen",
    "monitor", "shoes", "hat", "toy", "food", "camera", "clothes",
    "price tag", "sticker", "sample",
]
SCENE_VOCAB = [
    "person", "woman", "man", "hand", "face", "phone", "smartphone", "bottle",
    "cup", "drink", "product", "cosmetics", "lipstick", "box", "package",
    "bag", "shopping bag", "table", "chair", "screen", "monitor", "laptop",
    "microphone", "book", "paper", "card", "remote", "watch", "glasses",
    "food", "toy", "shoes", "hat", "camera", "plant", "window", "carton",
    "stand", "clothes", "price tag", "sticker", "sample", "warehouse",
]


def cache_dir(repo_id: str) -> Path:
    name = "models--" + repo_id.replace("/", "--")
    for root in MODELS_ROOTS:
        local = root / name
        if local.exists():
            return local
    return HF_HOME / ("models--" + repo_id.replace("/", "--"))


def resolve_model_dir(repo_id: str) -> Path:
    """Return a directory that transformers can load directly.

    Prefers a clean local copy (work/models), otherwise resolves the HF cache
    to its default snapshot directory (transformers 5.x can't load the cache
    root directly).
    """
    d = cache_dir(repo_id)
    if (d / "config.json").exists() or (d / "model.safetensors").exists() \
            or (d / "pytorch_model.bin").exists() or (d / "model.bin").exists():
        return d
    refs = d / "refs" / "main"
    if refs.exists():
        snap = d / "snapshots" / refs.read_text().strip()
        if snap.exists():
            return snap
    return d


def ffmpeg(args, **kw):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
        check=True, **kw,
    )


def video_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ]).decode().strip()
    return float(out)


def video_meta(path: Path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_name,codec_type,width,height,r_frame_rate", "-of", "json", str(path),
    ])
    streams = json.loads(out).get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), {})
    fps_str = vs.get("r_frame_rate", "0/1")
    num, den = (float(x) for x in fps_str.split("/"))
    fps = round(num / den, 3) if den else 0.0
    w, h = vs.get("width") or 0, vs.get("height") or 0
    return {
        "codec": vs.get("codec_name"),
        "width": w, "height": h, "fps": fps,
        "orientation": "horizontal" if w >= h else "vertical",
    }


def extract_frames(video: Path, outdir: Path, fps: float = 1.0):
    outdir.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
            str(outdir / "f_%05d.png")])
    return sorted(outdir.glob("f_*.png"))


def detect_scenes(video: Path, threshold: float = 0.25):
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video), "-vf",
         f"select='gt(scene,{threshold})',showinfo", "-vsync", "vfr",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    ts = [0.0]
    for m in re.finditer(r"pts_time:([\d.]+)", p.stderr):
        ts.append(float(m.group(1)))
    return sorted(set(round(x, 2) for x in ts))


def extract_audio(video: Path, wav: Path):
    ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-y", str(wav)])


def get_device(allow_mps: bool = True):
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenVocabDetector:
    def __init__(self):
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor
        d = resolve_model_dir("IDEA-Research/grounding-dino-base")
        self.processor = GroundingDinoProcessor.from_pretrained(d, local_files_only=True)
        self.device = get_device()
        try:
            self.model = GroundingDinoForObjectDetection.from_pretrained(
                d, local_files_only=True).to(self.device)
            with torch.no_grad():
                self.model.eval()
        except Exception as e:
            print(f"[detector] MPS load failed ({e}); fallback CPU")
            self.device = "cpu"
            self.model = GroundingDinoForObjectDetection.from_pretrained(
                d, local_files_only=True).to(self.device)
            self.model.eval()

    def detect(self, image: Image.Image, prompts, box_threshold=0.24, text_threshold=0.24):
        text = " . ".join(prompts)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        except RuntimeError:
            if self.device == "cpu":
                raise
            print("[detector] MPS inference failed; fallback CPU")
            self.device = "cpu"
            self.model = self.model.to("cpu")
            inputs = self.processor(images=image, text=text, return_tensors="pt").to("cpu")
            with torch.no_grad():
                outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=[prompts],
        )[0]
        labels = results.get("text_labels")
        if labels is None:
            labels = [prompts[i] for i in results["labels"]]
        return {
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
            "labels": labels,
        }


class CLIPEmbedder:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        d = resolve_model_dir("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(d, local_files_only=True)
        self.device = get_device()
        self.model = CLIPModel.from_pretrained(d, local_files_only=True).to(self.device)
        self.model.eval()

    def embed_images(self, images):
        if not images:
            return np.zeros((0, 512), dtype=np.float32)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        emb = getattr(out, "pooler_output", out)
        return F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)

    def embed_texts(self, texts):
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        emb = getattr(out, "pooler_output", out)
        return F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)


class Captioner:
    def __init__(self):
        self.available = False
        self.device = get_device()
        d = resolve_model_dir("Salesforce/blip-image-captioning-base")
        if not d.exists():
            print("[captioner] BLIP not downloaded; use --download-models first")
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.processor = BlipProcessor.from_pretrained(d, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(
            d, local_files_only=True).to(self.device)
        self.model.eval()
        self.available = True

    def caption(self, image: Image.Image) -> str:
        if not self.available:
            return ""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=48)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class TextEmbedder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.dim = 384
        d = resolve_model_dir("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if not d.exists():
            print("[text] multilingual model missing; fallback to English MiniLM")
            d = resolve_model_dir("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(d, local_files_only=True)
        self.device = get_device()
        self.model = AutoModel.from_pretrained(d, local_files_only=True).to(self.device)
        self.model.eval()

    def embed(self, texts):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=128,
            return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        mean = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(mean, dim=-1).cpu().numpy().astype(np.float32)


class WhisperASR:
    def __init__(self, size="tiny"):
        from faster_whisper import WhisperModel
        d = resolve_model_dir(f"Systran/faster-whisper-{size}")
        self.model = WhisperModel(str(d), device="cpu", compute_type="int8")

    def transcribe(self, wav: Path):
        segments, _ = self.model.transcribe(str(wav), vad_filter=True)
        return [(round(s.start, 2), round(s.end, 2), s.text.strip())
                for s in segments if s.text.strip()]


def crop_with_margin(img: Image.Image, box, margin=0.35, min_side=24):
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return None
    mx, my = w * margin, h * margin
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(img.width, x1 + mx), min(img.height, y1 + my)
    if x1 - x0 < min_side or y1 - y0 < min_side:
        return None
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aarea = (ax1 - ax0) * (ay1 - ay0)
    barea = (bx1 - bx0) * (by1 - by0)
    return inter / (aarea + barea - inter)


def make_contact_sheet(track_records, thumbs_dir, out_png: Path, cols=4, cell=200):
    if not track_records:
        Image.new("RGB", (cols * cell, cell), (20, 20, 24)).save(out_png)
        return
    rows = (len(track_records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 26)), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, rec in enumerate(track_records):
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + 26)
        thumb = thumbs_dir / f"track_{rec['track_id']}.jpg"
        if thumb.exists():
            im = Image.open(thumb).resize((cell - 8, cell - 8))
            sheet.paste(im, (x + 4, y + 4))
        label = f"#{rec['track_id']} {rec['label']} {rec['t0']}-{rec['t1']}s {rec['tier_en']}"
        draw.text((x + 4, y + cell - 2), label[:42], fill=(240, 240, 240), font=font)
    sheet.save(out_png, quality=88)


def triage_video(video: Path, duration: float, asr="tiny"):
    """快速分诊：语音密度 + 对象密度 -> 决定建纯 ASR 索引还是完整 AVIS 索引。

    成本约 20 秒：tiny ASR 一次 + 4-8 帧检测探测。
    """
    probe_n = min(8, max(4, int(duration / 10)))
    speech_coverage, transcript_chars = 0.0, 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if asr != "none":
            try:
                wav = tmp / "audio.wav"
                extract_audio(video, wav)
                segs = WhisperASR("tiny").transcribe(wav)
                speech_coverage = min(1.0, sum(e - s for s, e, _ in segs) / max(duration, 1.0))
                transcript_chars = sum(len(t) for _, _, t in segs)
            except Exception:
                speech_coverage = 0.0
        probe_fps = probe_n / duration
        frames = extract_frames(video, tmp / "probe", probe_fps)
        frames = frames[:probe_n]
        det = OpenVocabDetector()
        total, labels = 0, set()
        for f in frames:
            img = Image.open(f).convert("RGB")
            res = det.detect(img, DEFAULT_PROMPTS)
            for score, label in zip(res["scores"], res["labels"]):
                if score >= 0.22:
                    total += 1
                    labels.add(str(label))
    object_density = total / max(len(frames), 1)
    if asr == "none":
        mode, reason = "avis_full", "asr_disabled"
    elif speech_coverage >= 0.5 and object_density <= 1.5:
        mode, reason = "asr_only", "speech_dominant"
    elif speech_coverage < 0.3 and object_density >= 1.0:
        mode, reason = "avis_full", "visual_dominant"
    else:
        mode, reason = "avis_full", "balanced"
    return {
        "mode": mode, "reason": reason,
        "speech_coverage": round(speech_coverage, 3),
        "transcript_chars": transcript_chars,
        "object_density": round(object_density, 2),
        "labels_probed": sorted(labels),
        "probe_frames": len(frames),
    }


def build_asr_only(video: Path, out_dir: Path, asr, duration, triage_info):
    """纯 ASR 索引：只转写 + 场景边界，不检测、不嵌入视觉。"""
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[1/3] ASR transcription ...")
    wav = out_dir / "audio.wav"
    extract_audio(video, wav)
    model = WhisperASR(asr)
    asr_records = [{"start": s, "end": e, "text": t}
                   for s, e, t in model.transcribe(wav)]
    print(f"      {len(asr_records)} segments")
    print("[2/3] text embeddings + scene boundaries ...")
    text = TextEmbedder()
    asr_embs = [text.embed([r["text"]])[0] for r in asr_records]
    boundaries = detect_scenes(video, 0.15)
    min_segments = max(2, int(duration // 12))
    if len(boundaries) < min_segments:
        boundaries = [round(i * (duration / min_segments), 2)
                      for i in range(min_segments + 1)]
    scene_records = [{"scene_id": k, "t0": round(st, 2),
                      "t1": round(boundaries[k + 1] if k + 1 < len(boundaries) else duration, 2),
                      "keyframe_ts": round(st, 2), "caption": "", "tags": []}
                     for k, st in enumerate(boundaries)]
    print("[3/3] writing index ...")
    timeline_entries = []
    for i, (r, v) in enumerate(zip(asr_records, asr_embs)):
        timeline_entries.append({**r, "type": "asr", "emb_type": "text",
                                 "emb_index": i, "emb": [round(float(x), 4) for x in v]})
    np.savez_compressed(
        out_dir / "embeddings.npz",
        obj_emb=np.zeros((0, 512), np.float32),
        obj_text_emb=np.zeros((0, 384), np.float32),
        scene_emb=np.zeros((0, 384), np.float32),
        frame_emb=np.zeros((0, 512), np.float32),
        asr_emb=np.stack(asr_embs) if asr_embs else np.zeros((0, 384), np.float32),
        vocab_emb=np.zeros((0, 512), np.float32),
    )
    (out_dir / "objects.jsonl").write_text("", encoding="utf-8")
    (out_dir / "scenes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scene_records) + "\n",
        encoding="utf-8")
    (out_dir / "asr.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in asr_records) + "\n",
        encoding="utf-8")
    (out_dir / "timeline.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in timeline_entries) + "\n",
        encoding="utf-8")
    make_contact_sheet([], out_dir / "thumbs", out_dir / "tracks.png")
    build_sec = round(time.time() - t_start, 1)
    index_bytes = sum(f.stat().st_size for f in out_dir.glob("*") if f.is_file())
    manifest = {
        "semantic_layer_version": "0.2.1",
        "mode": triage_info,
        "video": {"path": str(video), "duration_sec": duration},
        "stats": {
            "frames": 0, "detections": 0, "tracks": 0, "appearances": 0,
            "scenes": len(scene_records), "asr_segments": len(asr_records),
            "tiers": {}, "build_sec": build_sec, "index_bytes": index_bytes,
        },
        "models": {"asr": f"Systran/faster-whisper-{asr}",
                   "text": "paraphrase-multilingual-MiniLM-L12-v2"},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    avis_json = {
        "avis_version": "0.1.0",
        "video": {**video_meta(video), "path": str(video), "name": video.name,
                  "stem": video.stem, "duration": duration, "format": "asr-only"},
        "signals": {
            "asr": {"path": "asr.jsonl", "segments": len(asr_records), "model": asr},
            "scenes": {"path": "scenes.jsonl", "boundaries": len(scene_records)},
            "objects": {"available": False, "reason": "asr_only"},
        },
        "timeline": {"duration_sec": duration},
        "clips": [],
    }
    (out_dir / "avis.json").write_text(
        json.dumps(avis_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {build_sec}s | asr_only index {index_bytes/1024:.1f} KB | "
          f"{len(asr_records)} ASR segments")


def build(video: Path, out_dir: Path, fps=1.0, prompts=None, asr="large-v3",
          max_frames=400, mode="auto"):
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    thumbs_dir = out_dir / "thumbs"
    prompts = prompts or DEFAULT_PROMPTS
    duration = video_duration(video)

    triage_info = {"mode": mode, "reason": "user_requested",
                   "speech_coverage": None, "object_density": None}
    if mode == "auto":
        triage_info = triage_video(video, duration, asr)
        mode = triage_info["mode"]
        print(f"[0/7] triage -> {mode} ({triage_info['reason']}) "
              f"speech={triage_info['speech_coverage']} "
              f"obj_density={triage_info['object_density']} "
              f"probe={triage_info['probe_frames']} frames")
    if mode == "asr_only":
        build_asr_only(video, out_dir, asr, duration, triage_info)
        return

    print(f"[1/7] extract frames @{fps}fps ...")
    files = extract_frames(video, frames_dir, fps)
    if len(files) > max_frames:
        files = files[:max_frames]
    print(f"      {len(files)} frames, video {duration:.1f}s")

    print("[2/7] load models (GroundingDINO + CLIP) ...")
    detector = OpenVocabDetector()
    clip = CLIPEmbedder()

    print("[3/7] open-vocab detection ...")
    detections = {}  # frame_idx -> list of dict
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        res = detector.detect(img, prompts)
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            if score < 0.22:
                continue
            crop = crop_with_margin(img, box)
            if crop is None:
                continue
            detections.setdefault(i, []).append({
                "t": round(i / fps, 3), "box": [float(v) for v in box],
                "score": float(score), "label": str(label), "crop": crop,
            })
    n_det = sum(len(v) for v in detections.values())
    print(f"      {n_det} detections")

    print("[4/7] IoU tracking ...")
    tracks = {}   # tid -> {"dets": [...], "missing": int}
    active = {}   # tid -> last box
    next_tid = 1
    for i in sorted(detections):
        dets = detections[i]
        assigned = set()
        for det in dets:
            best_tid, best_iou = None, 0.15
            for tid, last_box in active.items():
                if tid in assigned:
                    continue
                iou = box_iou(det["box"], last_box)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is None:
                best_tid = next_tid
                next_tid += 1
                tracks[best_tid] = {"dets": [], "missing": 0}
                active[best_tid] = det["box"]
            tracks[best_tid]["dets"].append(det)
            tracks[best_tid]["missing"] = 0
            active[best_tid] = det["box"]
            assigned.add(best_tid)
        for tid in list(active.keys()):
            if tid not in assigned:
                tracks[tid]["missing"] += 1
                if tracks[tid]["missing"] > 3:
                    del active[tid]
    print(f"      {len(tracks)} tracks")

    print("[5/7] CLIP embeddings for every object detection ...")
    crop_list, det_pos = [], []
    for tid, tr in tracks.items():
        for det in tr["dets"]:
            crop_list.append(det["crop"])
            det_pos.append(tid)
    embs = clip.embed_images(crop_list)
    pos_counter = Counter(det_pos)
    cursor = 0
    for tid, tr in tracks.items():
        n = pos_counter[tid]
        chunk = embs[cursor:cursor + n]
        cursor += n
        tr["emb_list"] = chunk
        for det, e in zip(tr["dets"], chunk):
            det["emb"] = e

    print("[6/7] scenes + captions + ASR ...")
    boundaries = detect_scenes(video, 0.15)
    min_segments = max(2, int(duration // 12))
    if len(boundaries) < min_segments:
        boundaries = [round(i * (duration / min_segments), 2)
                      for i in range(min_segments + 1)]
    scene_records, scene_embs, frame_embs = [], [], []
    blip = Captioner()
    text = TextEmbedder()
    vocab_embs = clip.embed_texts(SCENE_VOCAB)

    def nearest_frame(ts):
        idx = min(len(files) - 1, max(0, int(round(ts * fps))))
        return Image.open(files[idx]).convert("RGB")

    for k, st in enumerate(boundaries):
        t1 = boundaries[k + 1] if k + 1 < len(boundaries) else duration
        key_img = nearest_frame(st)
        cap = blip.caption(key_img) if blip.available else ""
        kf_emb = clip.embed_images([key_img])[0]
        sims = kf_emb @ vocab_embs.T
        top3 = [SCENE_VOCAB[j] for j in np.argsort(sims)[-3:][::-1]]
        if not cap:
            cap = ", ".join(top3)
        scene_records.append({
            "scene_id": k, "t0": round(st, 2), "t1": round(t1, 2),
            "keyframe_ts": round(st, 2), "caption": cap, "tags": top3,
        })
        scene_embs.append(text.embed([cap])[0])
        frame_embs.append(kf_emb)

    asr_records, asr_embs = [], []
    if asr:
        wav = out_dir / "audio.wav"
        extract_audio(video, wav)
        model = WhisperASR(asr)
        for start, end, seg_text in model.transcribe(wav):
            asr_records.append({"start": start, "end": end, "text": seg_text})
            asr_embs.append(text.embed([seg_text])[0])
        print(f"      {len(scene_records)} scenes, {len(asr_records)} ASR segments")

    print("[7/7] tiering + appearances + main-object understanding ...")
    track_records, object_entries = [], []
    obj_emb_list, obj_text_emb_list = [], []
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    for tid, tr in sorted(tracks.items()):
        dets = sorted(tr["dets"], key=lambda d: d["t"])
        t_first, t_last = dets[0]["t"], dets[-1]["t"]
        dur = t_last - t_first + 1 / fps
        frac = dur / duration
        tier = "主角" if frac >= 0.10 else ("配角" if frac >= 0.03 else "龙套")
        tier_en = "main" if tier == "主角" else ("support" if tier == "配角" else "cameo")
        label = Counter(d["label"] for d in dets).most_common(1)[0][0]
        groups = []
        for d in dets:
            if groups and d["t"] - groups[-1][-1]["t"] <= 3.0:
                groups[-1].append(d)
            else:
                groups.append([d])
        track_records.append({
            "track_id": tid, "label": label, "tier": tier, "tier_en": tier_en,
            "t0": round(t_first, 2), "t1": round(t_last, 2),
            "duration_sec": round(dur, 2), "fraction": round(frac, 4),
            "detections": len(dets), "appearances": len(groups),
        })
        thumb = groups[0][len(groups[0]) // 2]["crop"]
        thumb.resize((192, 192)).save(thumbs_dir / f"track_{tid}.jpg", quality=85)
        for gi, g in enumerate(groups):
            g0, g1 = g[0]["t"], g[-1]["t"]
            rep = g[len(g) // 2]
            mean = np.mean(np.stack([d["emb"] for d in g]), axis=0)
            mean = mean / np.linalg.norm(mean)
            desc = label
            if tier == "主角" and blip.available:
                cap = blip.caption(rep["crop"])
                if cap:
                    desc = cap
            object_entries.append({
                "type": "object", "track_id": tid, "label": label, "tier": tier,
                "appearance": gi, "t0": round(g0, 2), "t1": round(g1, 2),
                "n_dets": len(g), "description": desc,
            })
            obj_emb_list.append(mean.astype(np.float32))
            obj_text_emb_list.append(text.embed([desc])[0])

    print("      writing timeline ...")
    timeline_entries = []
    for i, (entry, v, tv) in enumerate(zip(object_entries, obj_emb_list, obj_text_emb_list)):
        timeline_entries.append({**entry, "emb_type": "visual", "emb_index": i,
                                 "emb": [round(float(x), 4) for x in v]})
        timeline_entries.append({**entry, "emb_type": "text", "emb_index": i,
                                 "emb": [round(float(x), 4) for x in tv]})
    for i, (s, v) in enumerate(zip(scene_records, scene_embs)):
        timeline_entries.append({**s, "type": "scene", "emb_type": "text", "emb_index": i,
                                 "emb": [round(float(x), 4) for x in v]})
    for i, (s, v) in enumerate(zip(asr_records, asr_embs)):
        timeline_entries.append({**s, "type": "asr", "emb_type": "text", "emb_index": i,
                                 "emb": [round(float(x), 4) for x in v]})

    np.savez_compressed(
        out_dir / "embeddings.npz",
        obj_emb=np.stack(obj_emb_list) if obj_emb_list else np.zeros((0, 512), np.float32),
        obj_text_emb=np.stack(obj_text_emb_list) if obj_text_emb_list else np.zeros((0, 384), np.float32),
        scene_emb=np.stack(scene_embs) if scene_embs else np.zeros((0, 384), np.float32),
        frame_emb=np.stack(frame_embs) if frame_embs else np.zeros((0, 512), np.float32),
        asr_emb=np.stack(asr_embs) if asr_embs else np.zeros((0, 384), np.float32),
        vocab_emb=vocab_embs,
    )
    (out_dir / "objects.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in object_entries) + "\n",
        encoding="utf-8")
    (out_dir / "timeline.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in timeline_entries) + "\n",
        encoding="utf-8")
    (out_dir / "scenes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scene_records) + "\n",
        encoding="utf-8")
    (out_dir / "asr.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in asr_records) + "\n",
        encoding="utf-8")
    make_contact_sheet(track_records, thumbs_dir, out_dir / "tracks.png")

    build_sec = round(time.time() - t_start, 1)
    index_bytes = sum(f.stat().st_size for f in out_dir.glob("*") if f.is_file())
    timeline_bytes = (out_dir / "timeline.jsonl").stat().st_size
    meta = video_meta(video)
    manifest = {
        "semantic_layer_version": "0.2.1",
        "mode": triage_info,
        "video": {"path": str(video), "duration_sec": duration, "fps_sampled": fps},
        "stats": {
            "frames": len(files), "detections": n_det, "tracks": len(track_records),
            "appearances": len(object_entries), "scenes": len(scene_records),
            "asr_segments": len(asr_records),
            "tiers": dict(Counter(r["tier"] for r in track_records)),
            "build_sec": build_sec, "index_bytes": index_bytes,
            "timeline_bytes": timeline_bytes,
        },
        "models": {
            "detector": "IDEA-Research/grounding-dino-base",
            "visual": "openai/clip-vit-base-patch32",
            "captioner": "Salesforce/blip-image-captioning-base" if blip.available else "clip-vocab",
            "text": "paraphrase-multilingual-MiniLM-L12-v2",
            "asr": f"Systran/faster-whisper-{asr}",
        },
        "prompts": prompts,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    avis_json = {
        "avis_version": "0.1.0",
        "mode": triage_info,
        "video": {**meta, "path": str(video), "name": video.name, "stem": video.stem,
                  "duration": duration, "format": "semantic-layer-demo"},
        "signals": {
            "asr": {"path": "asr.jsonl", "segments": len(asr_records), "model": asr},
            "scenes": {"path": "scenes.jsonl", "boundaries": len(scene_records)},
            "objects": {
                "path": "objects.jsonl", "timeline": "timeline.jsonl",
                "embeddings": "embeddings.npz", "tracks": len(track_records),
                "appearances": len(object_entries),
                "tiers": dict(Counter(r["tier"] for r in track_records)),
            },
        },
        "timeline": {"duration_sec": duration},
        "clips": [],
    }
    (out_dir / "avis.json").write_text(
        json.dumps(avis_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {build_sec}s | index {index_bytes/1024:.1f} KB | "
          f"{len(track_records)} tracks -> {len(object_entries)} appearances, "
          f"{len(scene_records)} scenes, {len(asr_records)} ASR | "
          f"timeline {timeline_bytes/1024:.1f} KB")


def load_index(idx_dir: Path):
    manifest = json.loads((idx_dir / "manifest.json").read_text(encoding="utf-8"))
    objects = [json.loads(x) for x in (idx_dir / "objects.jsonl").read_text(encoding="utf-8").splitlines() if x]
    scenes = [json.loads(x) for x in (idx_dir / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if x]
    asr = [json.loads(x) for x in (idx_dir / "asr.jsonl").read_text(encoding="utf-8").splitlines() if x]
    emb = np.load(idx_dir / "embeddings.npz")
    return manifest, objects, scenes, asr, emb


def query(idx_dir: Path, text=None, image=None, n=5, merge_gap=3.0):
    manifest, objects, scenes, asr, emb = load_index(idx_dir)
    hits = []
    if image:
        clip = CLIPEmbedder()
        q = clip.embed_images([Image.open(image).convert("RGB")])[0]
        if len(objects) and len(emb["obj_emb"]):
            obj_scores = q @ emb["obj_emb"].T
            for rank in np.argsort(obj_scores)[::-1][:n]:
                r = objects[rank]
                hits.append({"layer": "objects", "score": round(float(obj_scores[rank]), 3),
                             "t0": r["t0"], "t1": r["t1"],
                             "text": f"{r['description']} (#{r['track_id']}, {r['tier']})"})
        if len(scenes) and len(emb["frame_emb"]):
            fr_scores = q @ emb["frame_emb"].T
            for rank in np.argsort(fr_scores)[::-1][:n]:
                s = scenes[rank]
                hits.append({"layer": "scenes", "score": round(float(fr_scores[rank]), 3),
                             "t0": s["t0"], "t1": s["t1"], "text": s["caption"]})
    if text:
        te = TextEmbedder()
        q = te.embed([text])[0]
        if len(objects) and len(emb["obj_text_emb"]):
            desc_scores = q @ emb["obj_text_emb"].T
            for rank in np.argsort(desc_scores)[::-1][:n]:
                r = objects[rank]
                hits.append({"layer": "objects", "score": round(float(desc_scores[rank]), 3),
                             "t0": r["t0"], "t1": r["t1"],
                             "text": f"{r['description']} (#{r['track_id']}, {r['tier']})"})
            labels = sorted({o["label"] for o in objects})
            if labels:
                label_embs = te.embed(labels)
                lab_scores = q @ label_embs.T
                for rank in np.argsort(lab_scores)[::-1][:n]:
                    label = labels[rank]
                    for r in [o for o in objects if o["label"] == label][:3]:
                        hits.append({"layer": "objects", "score": round(float(lab_scores[rank]), 3),
                                     "t0": r["t0"], "t1": r["t1"],
                                     "text": f"{label} (#{r['track_id']}, {r['tier']})"})
        if len(scenes) and len(emb["scene_emb"]):
            sc_scores = q @ emb["scene_emb"].T
            for rank in np.argsort(sc_scores)[::-1][:n]:
                s = scenes[rank]
                hits.append({"layer": "scenes", "score": round(float(sc_scores[rank]), 3),
                             "t0": s["t0"], "t1": s["t1"], "text": s["caption"]})
        if len(asr) and len(emb["asr_emb"]):
            asr_scores = q @ emb["asr_emb"].T
            for rank in np.argsort(asr_scores)[::-1][:n]:
                a = asr[rank]
                hits.append({"layer": "asr", "score": round(float(asr_scores[rank]), 3),
                             "t0": a["start"], "t1": a["end"], "text": a["text"]})

    hits.sort(key=lambda h: h["score"], reverse=True)
    merged = []
    for h in hits:
        if (merged and h["t0"] <= merged[-1]["t1"] + merge_gap
                and h["t1"] - merged[-1]["t0"] <= 15.0):
            m = merged[-1]
            m["t1"] = max(m["t1"], h["t1"])
            m["score"] = max(m["score"], h["score"])
            m["layers"] = m.get("layers", [m["layer"]]) + [h["layer"]]
            m["text"] += " | " + h["text"]
        else:
            h["layers"] = [h["layer"]]
            merged.append(h)
    merged.sort(key=lambda h: h["score"], reverse=True)
    print(f"\nquery: {text or image} (top {n}, merged {merge_gap}s)\n")
    for i, h in enumerate(merged[:n], 1):
        bar = "#" * int(h["score"] * 30)
        print(f"[{i}] {h['t0']:>7.1f}-{h['t1']:<7.1f}s score={h['score']:.3f} "
              f"layers={','.join(h['layers'])} {bar}")
        print(f"      {h['text'][:110]}")
    return merged[:n]


def _entry_hits(objects, scenes, asr, emb, text=None, image=None,
                topk=5, max_window=30.0):
    """逐条目候选（不做窗口合并）：对象出现片段 + 场景 + ASR 句子。"""
    hits = []

    def add(t0, t1, score, layer, txt):
        if t1 - t0 <= max_window:
            hits.append((float(t0), float(t1), float(score), layer, txt))

    if image:
        clip = CLIPEmbedder()
        q = clip.embed_images([Image.open(image).convert("RGB")])[0]
        if len(objects) and len(emb["obj_emb"]):
            for r in np.argsort(q @ emb["obj_emb"].T)[::-1][:topk]:
                o = objects[r]
                add(o["t0"], o["t1"], q @ emb["obj_emb"].T[r],
                    f"objects:{o['label']}", o["description"])
        if len(scenes) and len(emb["frame_emb"]):
            for r in np.argsort(q @ emb["frame_emb"].T)[::-1][:topk]:
                sc = scenes[r]
                add(sc["t0"], sc["t1"], q @ emb["frame_emb"].T[r], "scenes", sc["caption"])
    if text:
        te = TextEmbedder()
        q = te.embed([text])[0]
        if len(objects) and len(emb["obj_text_emb"]):
            scores = q @ emb["obj_text_emb"].T
            for r in np.argsort(scores)[::-1][:topk]:
                o = objects[r]
                add(o["t0"], o["t1"], scores[r], f"objects:{o['label']}", o["description"])
            labels = sorted({o["label"] for o in objects})
            le = te.embed(labels)
            ls = q @ le.T
            for r in np.argsort(ls)[::-1][:topk]:
                for o in [x for x in objects if x["label"] == labels[r]][:2]:
                    add(o["t0"], o["t1"], ls[r], f"label:{labels[r]}", o["description"])
        if len(scenes) and len(emb["scene_emb"]):
            scores = q @ emb["scene_emb"].T
            for r in np.argsort(scores)[::-1][:topk]:
                sc = scenes[r]
                add(sc["t0"], sc["t1"], scores[r], "scenes", sc["caption"])
        if len(asr) and len(emb["asr_emb"]):
            scores = q @ emb["asr_emb"].T
            for r in np.argsort(scores)[::-1][:topk]:
                a = asr[r]
                add(a["start"], a["end"], scores[r], "asr", a["text"])
    return hits


def plan_clips(idx_dir: Path, text=None, image=None, object_labels=None,
               require_speech=True, min_dur=5.0, max_dur=180.0,
               max_clips=10, align_scenes=True, merge_gap=1.5, max_window=30.0,
               target_dur=20.0, keywords=None):
    """剪辑规划：逐条目候选 -> 对象过滤 -> 语音过滤 -> 场景对齐 -> 剪辑清单。

    每条剪辑都带证据（命中的层、语音文本、语音覆盖秒数），
    方便 agent 或人工判断"这刀剪得对不对"。
    """
    manifest, objects, scenes, asr, emb = load_index(idx_dir)
    if not text and not image and not object_labels:
        raise ValueError("需要 text / image / object_labels 至少一个")
    keywords = [k for k in (keywords or []) if k]
    cands = _entry_hits(objects, scenes, asr, emb, text=text, image=image,
                        max_window=max_window)
    if object_labels:
        labels = set(object_labels)
        for o in objects:
            if o["label"] in labels and o["t1"] - o["t0"] <= max_window:
                cands.append((o["t0"], o["t1"], 0.5,
                              f"object:{o['label']}", o["description"]))
    # 按 (t0,t1) 去重，保留最高分
    best = {}
    for t0, t1, score, layer, txt in cands:
        key = (round(t0, 2), round(t1, 2))
        if key not in best or score > best[key][2]:
            best[key] = (t0, t1, score, layer, txt)
    cands = sorted(best.values(), key=lambda c: c[0])
    merged = []
    for t0, t1, score, layer, txt in cands:
        if (merged and t0 <= merged[-1][1] + merge_gap
                and t1 - merged[-1][0] <= max_dur):
            m = merged[-1]
            m[1] = max(m[1], t1)
            m[2] = max(m[2], score)
            m[3] = m[3] + "+" + layer
            m[4] = m[4] + " | " + txt
        else:
            merged.append([t0, t1, score, layer, txt])
    clips = []
    for t0, t1, score, layers, txt in merged:
        speech = sum(max(0.0, min(t1, a["end"]) - max(t0, a["start"])) for a in asr)
        if require_speech and speech < 1.0:
            continue
        if align_scenes:
            # 只在 1.5 秒内有场景边界才吸附，避免把短命中撑成整场
            bounds = [b for s in scenes for b in (s["t0"], s["t1"])]
            near0 = [b for b in bounds if abs(b - t0) <= 1.5]
            near1 = [b for b in bounds if abs(b - t1) <= 1.5]
            if near0:
                t0 = min(near0, key=lambda b: abs(b - t0))
            if near1:
                t1 = min(near1, key=lambda b: abs(b - t1))
            if t1 < t0:
                t0, t1 = t1, t0
        if t1 - t0 < min_dur:
            continue
        # 超长片段按场景边界切成目标时长内的子片段
        if t1 - t0 > target_dur and align_scenes and scenes:
            segs = [s for s in scenes if s["t0"] < t1 and s["t1"] > t0]
            cuts = sorted({t0} | {s["t0"] for s in segs if s["t0"] > t0} | {t1})
            start = cuts[0]
            for end in cuts[1:]:
                if end - start >= min_dur:
                    clips.append([start, end, score, layers, ""])
                if end - start >= target_dur * 0.5:
                    start = end
        else:
            clips.append([t0, t1, score, layers, ""])
    # 去重 + 证据补全
    best = {}
    for t0, t1, score, layers, _ in clips:
        key = (round(t0, 2), round(t1, 2))
        if key not in best or score > best[key][2]:
            best[key] = [t0, t1, score, layers]
    out = []
    for t0, t1, score, layers in sorted(best.values(), key=lambda c: -c[2]):
        speech = sum(max(0.0, min(t1, a["end"]) - max(t0, a["start"])) for a in asr)
        speech_text = " ".join(
            a["text"] for a in asr if a["start"] < t1 and a["end"] > t0)[:200]
        if keywords and not any(k in speech_text for k in keywords):
            continue
        out.append({
            "t0": round(t0, 2), "t1": round(t1, 2), "duration": round(t1 - t0, 2),
            "score": round(score, 3), "layers": layers,
            "speech_sec": round(speech, 2), "speech_text": speech_text,
        })
        if len(out) >= max_clips:
            break
    return {
        "clips": out,
        "stats": {
            "candidates": len(cands), "merged": len(merged), "clips": len(out),
            "require_speech": require_speech, "align_scenes": align_scenes,
            "max_window": max_window, "target_dur": target_dur,
            "keywords": keywords,
        },
    }


def info(idx_dir: Path):
    manifest, objects, scenes, asr, emb = load_index(idx_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    seen = {}
    for o in objects:
        seen.setdefault(o["track_id"], o)
    tracks = list(seen.values())
    print(f"\n{len(tracks)} tracks -> {len(objects)} appearances by tier:")
    for tier in ["主角", "配角", "龙套"]:
        rows = [t for t in tracks if t["tier"] == tier]
        print(f"  {tier}: {len(rows)}")
        for t in rows[:5]:
            print(f"    #{t['track_id']} {t['label']:<12} {t['t0']:>7.1f}-{t['t1']:<7.1f}s "
                  f"({t['duration_sec']}s, {t['fraction']*100:.1f}%)")


def cut(idx_dir: Path, out_dir: Path, plan, video_path=None, padding=0.0):
    """把 plan_clips 的剪辑清单真正切成 mp4 片段（ffmpeg -c copy，秒级）。"""
    manifest = json.loads((idx_dir / "manifest.json").read_text(encoding="utf-8"))
    video = Path(video_path) if video_path else Path(manifest["video"]["path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for i, c in enumerate(plan.get("clips", []), 1):
        t0 = max(0.0, c["t0"] - padding)
        dur = max(0.5, (c["t1"] + padding) - t0)
        out = out_dir / f"clip_{i:02d}_{t0:.0f}-{t0+dur:.0f}s.mp4"
        ffmpeg(["-ss", str(t0), "-i", str(video), "-t", str(dur),
                "-c", "copy", "-avoid_negative_ts", "make_zero", "-y", str(out)])
        made.append({"file": str(out), "t0": round(t0, 2),
                     "t1": round(t0 + dur, 2), "source": c.get("layers", ""),
                     "speech": c.get("speech_text", "")[:120]})
    return {"ok": True, "clips": made, "out_dir": str(out_dir)}


def main():
    ap = argparse.ArgumentParser(description="object-level video semantic layer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("video")
    b.add_argument("-o", "--out", required=True)
    b.add_argument("--fps", type=float, default=1.0)
    b.add_argument("--prompts", default=None)
    b.add_argument("--asr", default="large-v3",
                   help="tiny/base/large-v3 or none（默认 large-v3：token 不变，精度最高）")
    b.add_argument("--max-frames", type=int, default=400)
    b.add_argument("--mode", default="auto",
                   choices=["auto", "asr_only", "avis_full"],
                   help="auto: 快速分诊后自动选择")
    q = sub.add_parser("query")
    q.add_argument("idx")
    q.add_argument("--text", default=None)
    q.add_argument("--image", default=None)
    q.add_argument("-n", type=int, default=5)
    p = sub.add_parser("plan")
    p.add_argument("idx")
    p.add_argument("--text", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--objects", default=None, help="逗号分隔的对象标签，如 lipstick,price tag")
    p.add_argument("--no-speech", dest="require_speech", action="store_false", default=True)
    p.add_argument("--min-dur", type=float, default=5.0)
    p.add_argument("--max-dur", type=float, default=180.0)
    p.add_argument("--max-clips", type=int, default=10)
    p.add_argument("--max-window", type=float, default=30.0,
                   help="候选窗口最大时长（过滤全程跟踪的宽窗口）")
    p.add_argument("--target-dur", type=float, default=20.0,
                   help="目标片段时长，超长片段按场景边界切分")
    p.add_argument("--keywords", default=None,
                   help="关键词交叉验证（逗号分隔）：片段语音必须包含任一关键词")
    p.add_argument("--no-align-scenes", dest="align_scenes", action="store_false", default=True)
    i = sub.add_parser("info")
    i.add_argument("idx")
    c = sub.add_parser("cut")
    c.add_argument("idx")
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--plan", required=True, help="plan_clips 输出的 JSON 文件")
    c.add_argument("--video", default=None, help="视频路径（默认取 manifest 里的）")
    c.add_argument("--padding", type=float, default=0.0)
    args = ap.parse_args()

    if args.cmd == "build":
        prompts = args.prompts.split(" . ") if args.prompts else None
        build(Path(args.video), Path(args.out), fps=args.fps, prompts=prompts,
              asr=None if args.asr == "none" else args.asr,
              max_frames=args.max_frames, mode=args.mode)
    elif args.cmd == "query":
        if not args.text and not args.image:
            sys.exit("query needs --text or --image")
        query(Path(args.idx), text=args.text, image=args.image, n=args.n)
    elif args.cmd == "plan":
        plan = plan_clips(
            Path(args.idx), text=args.text, image=args.image,
            object_labels=args.objects.split(",") if args.objects else None,
            require_speech=args.require_speech, min_dur=args.min_dur,
            max_dur=args.max_dur, max_clips=args.max_clips,
            align_scenes=args.align_scenes, max_window=args.max_window,
            target_dur=args.target_dur,
            keywords=args.keywords.split(",") if args.keywords else None)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    elif args.cmd == "info":
        info(Path(args.idx))
    elif args.cmd == "cut":
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        res = cut(Path(args.idx), Path(args.out), plan,
                  video_path=args.video, padding=args.padding)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
