"""
最小化的“给节点的算法输出”

目标：
- 节点只需要知道：顺序（链式执行顺序）+ 自己负责的层范围（第几层到第几层）
- Worker/服务端只需要元数据（每层 compute_cost + layer_index）即可运行
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class LayerMeta:
    layer_index: int
    compute_cost: float


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    compute: float


def compute_based_layer_ranges(layers: List[LayerMeta], nodes: List[NodeInfo]) -> List[Tuple[int, int]]:
    """
    按算力切分：输出每个节点对应的 (start_layer, end_layer)，两端均包含。
    """
    if not layers or not nodes:
        return []

    layers_sorted = sorted(layers, key=lambda x: x.layer_index)
    total_cost = sum(max(0.0, l.compute_cost) for l in layers_sorted)
    total_compute = sum(max(0.0, n.compute) for n in nodes)

    # 退化：按层数均分
    if total_cost <= 0.0 or total_compute <= 0.0:
        total_layers = len(layers_sorted)
        base = total_layers // len(nodes)
        rem = total_layers % len(nodes)
        ranges: List[Tuple[int, int]] = []
        cursor = 0
        for i in range(len(nodes)):
            take = base + (1 if i < rem else 0)
            if take <= 0:
                last = layers_sorted[-1].layer_index
                ranges.append((last, last))
                continue
            start = cursor
            end = cursor + take - 1
            cursor += take
            ranges.append((layers_sorted[start].layer_index, layers_sorted[end].layer_index))
        return ranges

    targets = [total_cost * (max(0.0, n.compute) / total_compute) for n in nodes]

    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for i, target in enumerate(targets):
        if cursor >= len(layers_sorted):
            last = layers_sorted[-1].layer_index
            ranges.append((last, last))
            continue

        start_idx = cursor
        acc = 0.0

        while cursor < len(layers_sorted):
            acc += max(0.0, layers_sorted[cursor].compute_cost)
            cursor += 1

            if i == len(nodes) - 1:
                break

            if acc >= target and (cursor - start_idx) >= 1:
                break

        end_idx = max(start_idx, cursor - 1)
        ranges.append((layers_sorted[start_idx].layer_index, layers_sorted[end_idx].layer_index))

    return ranges


def megaphone_order(nodes: List[NodeInfo]) -> List[str]:
    """
    最小化：直接采用输入顺序作为链式顺序。
    """
    return [n.node_id for n in nodes]


def build_minimal_plan(model_id: str, layers: List[LayerMeta], nodes: List[NodeInfo]) -> Dict[str, Any]:
    """
    返回给节点的最小 plan（JSON 友好）。
    """
    ranges = compute_based_layer_ranges(layers, nodes)
    order = megaphone_order(nodes)

    assignments = []
    for node, (start_layer, end_layer) in zip(nodes, ranges):
        assignments.append(
            {
                "node_id": node.node_id,
                "layer_range": {"start_layer": int(start_layer), "end_layer": int(end_layer)},
            }
        )

    return {
        "plan_version": 1,
        "model_id": model_id,
        "order": order,
        "assignments": assignments,
    }

