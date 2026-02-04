"""
生成用于 Worker 规划的精简元数据（包含稳定 layer_index）。

说明：
- 这里不做 Worker 侧的 state_dict 读取，节点本地读取/提取即可。
- 输出只包含 Worker 规划所需字段：layer_index / compute_required（可扩展 name/num_params）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def estimate_compute_required(t: torch.Tensor) -> float:
    # 极简保守估算：按参数量线性估计（你后续可替换成更真实的按层类型估算）
    # 这里的单位是“相对算力”，用于分配比例，不绑定真实 GFLOPS。
    return float(t.numel())


def build_metadata(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    layer_names: List[str] = list(state_dict.keys())
    layers: List[Dict[str, Any]] = []

    for idx, name in enumerate(layer_names):
        t = state_dict[name]
        layers.append(
            {
                "layer_index": idx,
                "name": name,
                "compute_required": estimate_compute_required(t),
                "num_params": int(t.numel()),
            }
        )

    return {"layers": layers}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dict", required=True, help="本地 .pth 的 state_dict 路径")
    ap.add_argument("--out", required=True, help="输出 metadata.json 路径")
    args = ap.parse_args()

    state_path = Path(args.state_dict)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sd = torch.load(state_path, map_location="cpu")
    if not isinstance(sd, dict) or len(sd) == 0:
        raise RuntimeError("state_dict 为空或格式不正确")

    metadata = build_metadata(sd)
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入: {out_path} (layers={len(metadata['layers'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

