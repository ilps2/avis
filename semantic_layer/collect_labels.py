#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总所有标注包的结果（label 已填的 entry）。"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", help="标注包根目录")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    root = Path(args.packs)
    labeled, total = [], 0
    for labels_file in sorted(root.rglob("labels.json")):
        data = json.loads(labels_file.read_text(encoding="utf-8"))
        video = data["video"]
        rule = labels_file.parent.name
        for e in data["entries"]:
            total += 1
            if e.get("label") is not None:
                labeled.append({**e, "video": video, "rule": rule})
    out = Path(args.out)
    out.write_text(json.dumps({"labeled": labeled, "total": total},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    n1 = sum(1 for x in labeled if x["label"] == 1)
    print(f"{labeled}/{total} 已标注（其中 {n1} 条 = 值得剪）")
    print("saved:", out)


if __name__ == "__main__":
    main()
