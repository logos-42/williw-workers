#!/usr/bin/env python3
"""
Python 脚本：切分模型
被 Rust 模块调用
"""
import json
import sys
import argparse
import torch
from transformers import AutoModel
from pathlib import Path
from typing import Dict, Any

def split_model(model_name: str, model_path: str, 
               plan_file: str, output_dir: str, node_id: str) -> Dict[str, Any]:
    """根据方案切分模型"""
    # 加载切分方案
    with open(plan_file, 'r') as f:
        plan = json.load(f)
    
    layer_names = plan["layer_names"]
    
    # 加载模型
    model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
    state_dict = model.state_dict()
    
    # 提取本节点的层
    my_shard = {name: state_dict[name].clone() for name in layer_names}
    
    # 保存分片
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    shard_path = output_path / f"shard_{node_id}.pth"
    torch.save(my_shard, shard_path)
    
    total_params = sum(p.numel() for p in my_shard.values())
    shard_size_mb = sum(p.numel() * 4 for p in my_shard.values()) / (1024 * 1024)
    
    result = {
        "node_id": node_id,
        "shard_path": str(shard_path),
        "layer_names": layer_names,
        "total_params": total_params,
        "shard_size_mb": shard_size_mb
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-id", required=True)
    
    args = parser.parse_args()
    
    result = split_model(
        args.model_name,
        args.model_path,
        args.plan_file,
        args.output_dir,
        args.node_id
    )
    
    # 输出 JSON
    print(json.dumps(result, indent=2))
