#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download HF repos through hf-mirror with curl (bypass huggingface_hub redirect bug)."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "models"
MIRROR = "https://hf-mirror.com"

REPOS = {
    "blip": "Salesforce/blip-image-captioning-base",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "gdino": "IDEA-Research/grounding-dino-base",
}


def file_list(repo: str):
    url = f"{MIRROR}/api/models/{repo}"
    meta = json.loads(subprocess.check_output(
        ["curl", "-sL", "--max-time", "60", url]))
    return [s["rfilename"] for s in meta.get("siblings", [])]


def fetch_one(repo: str, rel: str):
    dest = ROOT / ("models--" + repo.replace("/", "--")) / rel
    if dest.exists() and dest.stat().st_size > 0:
        return f"skip  {rel}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{repo}/resolve/main/{rel}"
    subprocess.run(
        ["curl", "-sL", "--fail", "--retry", "2", "--max-time", "1800",
         "-o", str(dest), url],
        check=True,
    )
    return f"ok    {rel} ({dest.stat().st_size / 1e6:.1f} MB)"


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for key, repo in REPOS.items():
        if only and key != only:
            continue
        print(f"== {repo} ==")
        rels = file_list(repo)
        print(f"   {len(rels)} files")
        with ThreadPoolExecutor(max_workers=4) as ex:
            for out in ex.map(lambda rel: fetch_one(repo, rel), rels):
                print("   " + out)


if __name__ == "__main__":
    main()
