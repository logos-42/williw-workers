"""
节点侧 demo：模拟“从Worker拿到最小plan，然后按layer_range切分”

说明：
- 这里不做真实网络调用；直接构造元数据并调用 algorithms.minimal_plan 生成最小plan
- 然后用 algorithms.model_splitter.ModelSplitter.split_by_layer_ranges 做切分演示
"""

import json
from pathlib import Path

import torch

from algorithms.minimal_plan import LayerMeta, NodeInfo, build_minimal_plan
from algorithms.model_splitter import ModelSplitter


def _fake_state_dict(num_layers: int = 24):
    # 用线性层权重张量模拟每一“层”的参数
    sd = {}
    for i in range(num_layers):
        sd[f"layer_{i}.weight"] = torch.randn(64, 64)
    return sd


def main():
    model_id = "demo-model"

    # 1) 节点（或任意端）构造/读取元数据：每层的 compute_cost（这里用 numel 近似）
    sd = _fake_state_dict(num_layers=24)
    layers = [LayerMeta(layer_index=i, compute_cost=float(t.numel())) for i, t in enumerate(sd.values())]

    # 2) 节点信息（实际来自接口层/上报）
    nodes = [
        NodeInfo(node_id="node_a", compute=100.0),
        NodeInfo(node_id="node_b", compute=60.0),
        NodeInfo(node_id="node_c", compute=40.0),
    ]

    # 3) Worker算法输出（最小plan）
    minimal_plan = build_minimal_plan(model_id=model_id, layers=layers, nodes=nodes)

    # 4) 节点按 layer_range 切分
    splitter = ModelSplitter(output_dir="./model_shards")
    shards = splitter.split_by_layer_ranges(sd, minimal_plan["assignments"])

    # 5) 输出结果文件（节点最终需要保存/消费的内容）
    out = {
        "minimal_plan": minimal_plan,
        "shards": shards,
    }

    out_path = Path(__file__).resolve().parent / "demo_workflow_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()

