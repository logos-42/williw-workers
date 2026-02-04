"""
按 layer_index 范围切分本地 state_dict（节点侧执行）。

输入：
- --state-dict: 原始 .pth
- --start-layer / --end-layer: 规划结果给到的层范围（inclusive）
- --out: 输出 shard .pth

约定：
- layer_index 对应 state_dict.keys() 的顺序（与 metadata 生成保持一致）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dict", required=True)
    ap.add_argument("--start-layer", type=int, required=True)
    ap.add_argument("--end-layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    start = int(args.start_layer)
    end = int(args.end_layer)
    if start < 0 or end < start:
        raise ValueError("非法范围：start/end")

    sd = torch.load(args.state_dict, map_location="cpu")
    names = list(sd.keys())
    if end >= len(names):
        raise ValueError(f"end_layer 超界：end={end}, total={len(names)}")

    selected = names[start : end + 1]
    shard = {k: sd[k] for k in selected}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, out_path)
    print(f"写入: {out_path} (layers={len(selected)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

