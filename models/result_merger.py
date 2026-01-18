"""
结果集成模块
合并各节点的推理结果
"""
import torch
from typing import Dict, Any, List
import numpy as np


class ResultMerger:
    """结果集成器"""
    
    def merge(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        集成推理结果
        
        Args:
            inference_result: 推理结果字典（来自DistributedInferenceEngine）
        
        Returns:
            集成后的最终结果
        """
        if not inference_result.get('success'):
            return {
                'status': 'error',
                'message': inference_result.get('error', '推理失败')
            }
        
        final_output = inference_result.get('final_output')
        node_results = inference_result.get('node_results', [])
        
        # 处理最终输出
        if isinstance(final_output, torch.Tensor):
            # 转换为numpy或list（根据输出类型）
            if final_output.dim() == 0:
                # 标量
                result_value = final_output.item()
            elif final_output.dim() == 1:
                # 一维向量
                result_value = final_output.tolist()
            else:
                # 多维tensor
                result_value = final_output.detach().cpu().numpy().tolist()
        elif isinstance(final_output, dict):
            # 字典输出（如Transformer模型的多个输出）
            result_value = {}
            for key, value in final_output.items():
                if isinstance(value, torch.Tensor):
                    result_value[key] = value.detach().cpu().numpy().tolist()
                else:
                    result_value[key] = value
        else:
            # 其他类型
            result_value = final_output
        
        # 构建结果摘要
        result_summary = {
            'status': 'success',
            'output': result_value,
            'num_nodes': len(node_results),
            'nodes_used': [r['node_id'] for r in node_results],
            'inference_time_ms': inference_result.get('total_time', 0.0)
        }
        
        return result_summary
    
    def merge_parallel_results(self, node_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并并行推理结果（如MoE模型）
        
        Args:
            node_results: 各节点的结果列表
        
        Returns:
            合并后的结果
        """
        # 对于并行推理，需要合并多个输出
        # 这里简化处理：平均或加权平均
        
        outputs = []
        for node_result in node_results:
            output = node_result.get('output')
            if output is not None:
                if isinstance(output, torch.Tensor):
                    outputs.append(output)
        
        if len(outputs) == 0:
            return {
                'status': 'error',
                'message': '没有有效的结果'
            }
        
        # 平均多个输出
        if len(outputs) > 1:
            merged_output = torch.mean(torch.stack(outputs), dim=0)
        else:
            merged_output = outputs[0]
        
        return {
            'status': 'success',
            'output': merged_output.detach().cpu().numpy().tolist(),
            'num_nodes': len(outputs)
        }
