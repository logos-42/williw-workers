"""
完整模型切分模块
基于PyTorch state_dict按层切分模型
"""
from typing import Dict, List, Any
from pathlib import Path
import torch
import numpy as np


class ModelSplitter:
    """完整模型切分器"""
    
    def __init__(self, output_dir: str = "./model_shards"):
        """
        初始化模型切分器
        
        Args:
            output_dir: 分片输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def split_by_layers(self,
                       state_dict: Dict[str, torch.Tensor],
                       nodes: List,
                       split_strategy: str = 'equal') -> List[Dict[str, Any]]:
        """
        按层切分模型
        
        Args:
            state_dict: PyTorch模型state_dict
            nodes: 节点列表
            split_strategy: 切分策略（'equal'均匀切分, 'compute'按算力切分）
        
        Returns:
            模型分片列表
        """
        print(f"\n=== 模型切分开始 ===")
        layer_names = list(state_dict.keys())
        total_layers = len(layer_names)
        num_nodes = len(nodes)
        
        print(f"总层数: {total_layers}")
        print(f"节点数: {num_nodes}")
        
        if total_layers == 0:
            print("警告: 模型没有层")
            return []
        
        if num_nodes == 0:
            print("警告: 没有可用节点")
            return []
        
        # 确定每节点层数分配
        if split_strategy == 'equal':
            layer_assignments = self._equal_split(total_layers, num_nodes)
        elif split_strategy == 'compute':
            compute_powers = [self._estimate_node_compute(n) for n in nodes]
            layer_assignments = self._compute_based_split(total_layers, compute_powers)
        else:
            layer_assignments = self._equal_split(total_layers, num_nodes)
        
        # 创建分片
        shards = []
        layer_idx = 0
        
        for i, node in enumerate(nodes):
            num_layers = layer_assignments[i]
            node_layers = layer_names[layer_idx:layer_idx + num_layers]
            
            # 提取对应层的参数
            shard_state_dict = {name: state_dict[name].clone() for name in node_layers}
            
            # 保存分片
            shard_path = self.output_dir / f"shard_{getattr(node, 'node_id', i)}.pth"
            torch.save(shard_state_dict, shard_path)
            
            shards.append({
                'node_id': getattr(node, 'node_id', f'node_{i}'),
                'node_index': i,
                'shard_path': str(shard_path),
                'layer_names': node_layers,
                'layer_indices': list(range(layer_idx, layer_idx + num_layers)),
                'layer_count': len(node_layers),
                'total_params': sum(p.numel() for p in shard_state_dict.values()),
                'shard_size_mb': sum(p.numel() * 4 for p in shard_state_dict.values()) / (1024 * 1024)
            })
            
            print(f"节点{i} ({getattr(node, 'node_id', 'unknown')}): "
                  f"{len(node_layers)}层, {shards[-1]['total_params']}参数, "
                  f"{shards[-1]['shard_size_mb']:.2f} MB")
            
            layer_idx += num_layers
        
        print(f"模型切分完成，共{len(shards)}个分片")
        print("=== 模型切分完成 ===\n")
        
        return shards
    
    def _equal_split(self, total_layers: int, num_nodes: int) -> List[int]:
        """均匀切分"""
        base = total_layers // num_nodes
        remainder = total_layers % num_nodes
        
        assignments = [base] * num_nodes
        for i in range(remainder):
            assignments[i] += 1
        
        return assignments
    
    def _compute_based_split(self, total_layers: int, compute_powers: List[float]) -> List[int]:
        """按算力切分"""
        total_compute = sum(compute_powers)
        if total_compute == 0:
            return self._equal_split(total_layers, len(compute_powers))
        
        # 按算力比例分配
        ratios = [cp / total_compute for cp in compute_powers]
        assignments = [int(total_layers * r) for r in ratios]
        
        # 确保所有层都被分配
        assigned = sum(assignments)
        if assigned < total_layers:
            remainder = total_layers - assigned
            # 分配给算力最高的节点
            sorted_indices = sorted(range(len(compute_powers)), 
                                  key=lambda i: compute_powers[i], reverse=True)
            for i in range(remainder):
                assignments[sorted_indices[i]] += 1
        elif assigned > total_layers:
            # 如果分配过多，从算力最低的节点减少
            sorted_indices = sorted(range(len(compute_powers)), 
                                  key=lambda i: compute_powers[i])
            excess = assigned - total_layers
            for i in range(excess):
                if assignments[sorted_indices[i]] > 1:
                    assignments[sorted_indices[i]] -= 1
        
        return assignments
    
    def _estimate_node_compute(self, node) -> float:
        """估算节点算力"""
        if not getattr(node, 'gpu_available', False):
            cpu_cores = getattr(node, 'cpu_cores', 4)
            return cpu_cores * 10.0
        
        gpu_name = getattr(node, 'gpu_name', '').lower()
        gpu_compute_map = {
            'rtx 4090': 80000.0,
            'rtx 4080': 50000.0,
            'rtx 3090': 36000.0,
            'rtx 3080': 30000.0,
            'a100': 312000.0,
        }
        
        base_compute = 5000.0
        for gpu_key, compute in gpu_compute_map.items():
            if gpu_key in gpu_name:
                base_compute = compute
                break
        
        return base_compute
