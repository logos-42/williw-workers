"""
离线模拟 Rust Worker 的 /api/process 输出（用于 demo/回归测试）。

输入：demo_process_request.json（或任意同结构 JSON）
输出：demo_process_response.json（精简：仅层范围 + 顺序）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def equal_split(num_layers: int, nodes: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    n = len(nodes)
    base, rem = divmod(num_layers, n)
    out: List[Tuple[str, int, int]] = []
    start = 0
    for i, node in enumerate(nodes):
        cnt = base + (1 if i < rem else 0)
        if cnt <= 0:
            continue
        end = start + cnt - 1
        out.append((node["node_id"], start, end))
        start = end + 1
    if out and out[-1][2] != num_layers - 1:
        node_id, s, _ = out[-1]
        out[-1] = (node_id, s, num_layers - 1)
    return out


def compute_contiguous_split(layers: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    num_layers = len(layers)
    total_compute = sum(max(0.0, float(l.get("compute_required", 0.0))) for l in layers)
    node_total = sum(max(0.0, float(n.get("compute_power", 0.0))) for n in nodes)
    if total_compute <= 0.0 or node_total <= 0.0:
        return equal_split(num_layers, nodes)

    prefix = [0.0]
    for l in layers:
        prefix.append(prefix[-1] + max(0.0, float(l.get("compute_required", 0.0))))

    targets: List[float] = []
    acc = 0.0
    for n in nodes:
        acc += max(0.0, float(n.get("compute_power", 0.0))) / node_total * total_compute
        targets.append(acc)
    targets[-1] = total_compute

    out: List[Tuple[str, int, int]] = []
    start_idx = 0
    for i, node in enumerate(nodes):
        if start_idx >= num_layers:
            break
        target = targets[i]
        end_idx = start_idx
        while end_idx + 1 < len(prefix) and prefix[end_idx + 1] < target:
            end_idx += 1
        if i < len(nodes) - 1 and end_idx == num_layers - 1 and start_idx < num_layers - 1:
            end_idx = num_layers - 2
        out.append((node["node_id"], start_idx, end_idx))
        start_idx = end_idx + 1

    if out and out[-1][2] != num_layers - 1:
        node_id, s, _ = out[-1]
        out[-1] = (node_id, s, num_layers - 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="node_client/demo_process_request.json")
    ap.add_argument("--out", dest="outp", default="node_client/demo_process_response.json")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    req = json.loads(inp.read_text(encoding="utf-8"))
    model_id = req["model_id"]
    layers = req["metadata"]["layers"]
    nodes = req["nodes"]
    strategy = req.get("strategy") or "compute"

    if strategy == "equal":
        triples = equal_split(len(layers), nodes)
    else:
        triples = compute_contiguous_split(layers, nodes)

    resp = {
        "model_id": model_id,
        "num_layers": len(layers),
        "plan": {
            "assignments": [
                {"node_id": node_id, "start_layer": s, "end_layer": e} for (node_id, s, e) in triples
            ],
            "execution_order": [node_id for (node_id, _, _) in triples],
        },
    }

    outp.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resp, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

