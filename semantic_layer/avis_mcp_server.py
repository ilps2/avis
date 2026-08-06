#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AVIS MCP Server — agent 视频理解的 "ffmpeg" 原型

暴露 4 个工具给任何 MCP 客户端（Claude/Codex/Cursor 等 agent）：

  encode_video     视频 -> 语义层索引（一次编码，本地，零 token）
  search_index     文字/参考图查询 -> 时间轴命中窗口（免费检索）
  understand_video 查询 -> 命中片段 -> 本地 VLM 理解 -> 结论（两段式）
  index_info       索引统计

用法：
  python3 avis_mcp_server.py          # stdio 模式（MCP 默认）
"""

import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

import semantic_layer as sl

mcp = FastMCP("avis")

_vlm = {"model": None, "tokenizer": None}


def _silent(fn, *args, **kwargs):
    with contextlib.redirect_stdout(sys.stderr):
        return fn(*args, **kwargs)


def _load_vlm():
    if _vlm["model"] is None:
        from mlx_lm import load
        qdir = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-Instruct-4bit/snapshots"
        rev = next(p for p in qdir.iterdir() if p.is_dir())
        _vlm["model"], _vlm["tokenizer"] = load(str(rev))
    return _vlm["model"], _vlm["tokenizer"]


def _collect_evidence(idx_dir: Path, hits):
    def read(name):
        return [json.loads(x) for x in (idx_dir / name).read_text(encoding="utf-8").splitlines() if x]
    scenes, objects, asr = read("scenes.jsonl"), read("objects.jsonl"), read("asr.jsonl")
    evidence = {"scenes": [], "objects": [], "asr": []}
    for h in hits:
        t0, t1 = h["t0"], h["t1"]
        for s in scenes:
            if s["t0"] <= t1 and s["t1"] >= t0:
                evidence["scenes"].append(s)
        for o in objects:
            if o["t0"] <= t1 and o["t1"] >= t0:
                evidence["objects"].append(o)
        for a in asr:
            if a["start"] <= t1 and a["end"] >= t0:
                evidence["asr"].append(a)
    # 去重并截断
    for k in evidence:
        seen, out = set(), []
        for item in evidence[k]:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(item)
        evidence[k] = out[:15]
    return evidence


def _fmt_evidence(ev):
    lines = []
    lines.append("【命中场景】")
    lines += [f"[{s['t0']}-{s['t1']}s] {s['caption']}" for s in ev["scenes"][:8]]
    lines.append("【命中对象】")
    lines += [f"[{o['t0']}-{o['t1']}s] {o['label']}({o['tier']}): {o['description']}" for o in ev["objects"][:12]]
    lines.append("【命中语音】")
    lines += [f"[{a['start']}-{a['end']}s] {a['text']}" for a in ev["asr"][:10]]
    return "\n".join(lines)


@mcp.tool()
def encode_video(video_path: str, out_dir: str, fps: float = 1.0,
                 asr: str = "large-v3", max_frames: int = 400,
                 mode: str = "auto") -> str:
    """把视频编码成语义层索引（对象+场景+ASR+统一时间轴），一次编码、无限次免 token 查询。

    Args:
        video_path: 视频文件路径
        out_dir: 索引输出目录
        fps: 抽帧率（默认 1）
        asr: whisper 档位 tiny/base/large-v3/none（默认 large-v3：token 不变，精度最高）
        max_frames: 最多处理帧数
        mode: auto（快速分诊自动选择）/ asr_only / avis_full
    """
    t0 = time.time()
    _silent(sl.build, Path(video_path), Path(out_dir), fps=fps,
            asr=None if asr == "none" else asr, max_frames=max_frames, mode=mode)
    manifest = json.loads((Path(out_dir) / "manifest.json").read_text(encoding="utf-8"))
    return json.dumps({"ok": True, "build_seconds": round(time.time() - t0, 1),
                       "mode": manifest.get("mode"),
                       **manifest["stats"]}, ensure_ascii=False, indent=2)


@mcp.tool()
def search_index(idx_dir: str, text: str | None = None, image: str | None = None,
                 top_k: int = 5) -> str:
    """在语义层索引里做免 token 检索，返回带时间戳的命中窗口。

    Args:
        idx_dir: 语义层索引目录
        text: 中文/英文查询词
        image: 参考图路径（图搜图，稀有对象建议用这个）
        top_k: 返回条数
    """
    if not text and not image:
        return json.dumps({"error": "需要 text 或 image"}, ensure_ascii=False)
    hits = _silent(sl.query, Path(idx_dir), text=text, image=image, n=top_k)
    return json.dumps({"ok": True, "hits": hits}, ensure_ascii=False, indent=2)


@mcp.tool()
def understand_video(idx_dir: str, query: str, top_k: int = 5,
                     max_tokens: int = 300) -> str:
    """两段式视频理解：先在语义层定位命中窗口（免费），再把命中片段交给本地 VLM 理解。

    Args:
        idx_dir: 语义层索引目录
        query: 要理解的问题
        top_k: 检索窗口数
        max_tokens: VLM 输出上限
    """
    t0 = time.time()
    hits = _silent(sl.query, Path(idx_dir), text=query, n=top_k)
    ev = _collect_evidence(Path(idx_dir), hits)
    payload = (f"以下是视频中与「{query}」相关的命中片段（已定位到时间轴）：\n"
               + _fmt_evidence(ev)
               + f"\n请用中文回答：{query}")
    model, tokenizer = _load_vlm()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": payload}], tokenize=False, add_generation_prompt=True)
    pt = len(tokenizer.encode(prompt))
    out = _silent(generate, model, tokenizer, prompt=prompt,
                  max_tokens=max_tokens, sampler=make_sampler(temp=0.3))
    ot = len(tokenizer.encode(out)) if out else 0
    return json.dumps({
        "ok": True, "query": query,
        "answer": out,
        "tokens": {"prompt": pt, "output": ot},
        "seconds": round(time.time() - t0, 1),
        "hits": hits,
        "evidence": ev,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def index_info(idx_dir: str) -> str:
    """查看语义层索引的统计信息（对象/场景/ASR/大小）。"""
    idx = Path(idx_dir)
    manifest = json.loads((idx / "manifest.json").read_text(encoding="utf-8"))
    timeline_bytes = (idx / "timeline.jsonl").stat().st_size if (idx / "timeline.jsonl").exists() else 0
    return json.dumps({"ok": True, **manifest["stats"],
                       "timeline_bytes": timeline_bytes}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
