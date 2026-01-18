"""
模型算力估算模块
基于模型结构和参数数量估算推理所需的算力
采用保守估算策略：可以算多，不能算少
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np


class ComputeEstimator:
    """模型算力估算器（保守估算）"""
    
    # 不同操作的算力系数（GFLOPS per operation）
    OPERATION_COSTS = {
        'conv2d': 2.0,      # 卷积：每参数每次推理约2 GFLOPS
        'linear': 2.0,      # 全连接：每参数每次推理约2 GFLOPS
        'attention': 4.0,   # 注意力：每参数每次推理约4 GFLOPS（矩阵乘更复杂）
        'layernorm': 1.0,   # 层归一化：每参数约1 GFLOPS
        'embedding': 0.5,   # 嵌入层：相对简单，0.5 GFLOPS
        'activation': 0.1,  # 激活函数：0.1 GFLOPS
        'pooling': 0.2,     # 池化：0.2 GFLOPS
    }
    
    # 激活值计算开销系数（额外的内存访问和计算）
    ACTIVATION_OVERHEAD = 1.5  # 激活值计算额外开销50%
    
    # 内存访问开销系数
    MEMORY_ACCESS_OVERHEAD = 1.3  # 内存访问额外开销30%
    
    # 安全系数（保守估算）
    SAFETY_FACTOR = 1.5  # 最终结果乘以1.5，确保不低估
    
    def __init__(self, batch_size: int = 1, sequence_length: int = 512):
        """
        初始化估算器
        
        Args:
            batch_size: 批次大小（影响激活值计算量）
            sequence_length: 序列长度（用于Transformer模型）
        """
        self.batch_size = batch_size
        self.sequence_length = sequence_length
    
    def estimate_from_state_dict(self, state_dict: Dict[str, torch.Tensor], 
                                 model_type: str = "auto") -> Dict[str, float]:
        """
        从state_dict估算模型算力需求
        
        Args:
            state_dict: 模型参数字典
            model_type: 模型类型（"cnn", "transformer", "rnn", "auto"）
        
        Returns:
            估算结果字典，包含：
            - total_compute: 总算力需求（GFLOPS）
            - memory_required: 内存需求（GB）
            - gpu_required: 是否需要GPU
            - estimated_latency: 估算延迟（ms）
        """
        # 统计参数数量
        total_params = sum(p.numel() for p in state_dict.values())
        
        # 识别模型类型
        if model_type == "auto":
            model_type = self._detect_model_type(state_dict)
        
        # 估算各类操作的算力
        compute_breakdown = self._estimate_by_layer_type(state_dict, model_type)
        
        # 总计算量
        base_compute = sum(compute_breakdown.values())
        
        # 激活值开销（前向传播需要计算和存储激活值）
        activation_compute = base_compute * self.ACTIVATION_OVERHEAD
        
        # 内存访问开销
        memory_compute = activation_compute * self.MEMORY_ACCESS_OVERHEAD
        
        # 应用安全系数（保守估算）
        total_compute = memory_compute * self.SAFETY_FACTOR
        
        # 估算内存需求
        param_memory = total_params * 4 / (1024**3)  # 假设float32，转换为GB
        # 激活值内存（估算）
        activation_memory = self._estimate_activation_memory(state_dict, model_type)
        total_memory = (param_memory + activation_memory) * 1.5  # 预留50%缓冲
        
        # 估算延迟（基于算力，假设1 GFLOPS = 1 ms，这是非常粗略的估算）
        # 实际延迟取决于硬件，这里只是粗略估算
        estimated_latency = total_compute / 100.0  # 假设100 GFLOPS设备需要这么多ms
        
        # 判断是否需要GPU（基于算力需求）
        gpu_required = total_compute > 10.0  # 超过10 GFLOPS建议使用GPU
        
        return {
            'total_compute': total_compute,  # GFLOPS
            'memory_required': total_memory,  # GB
            'gpu_required': gpu_required,
            'estimated_latency': estimated_latency,  # ms
            'total_params': total_params,
            'model_type': model_type,
            'compute_breakdown': compute_breakdown,
            'safety_factor_applied': self.SAFETY_FACTOR,
        }
    
    def _detect_model_type(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """自动检测模型类型"""
        keys = list(state_dict.keys())
        key_str = ' '.join(keys)
        
        # Transformer特征
        if any(k in key_str for k in ['attention', 'transformer', 'encoder', 'decoder', 'embedding']):
            return "transformer"
        
        # CNN特征
        if any(k in key_str for k in ['conv', 'bn', 'batch_norm', 'pool']):
            return "cnn"
        
        # RNN特征
        if any(k in key_str for k in ['lstm', 'rnn', 'gru', 'recurrent']):
            return "rnn"
        
        # 默认
        return "mlp"
    
    def _estimate_by_layer_type(self, state_dict: Dict[str, torch.Tensor], 
                                model_type: str) -> Dict[str, float]:
        """按层类型估算算力"""
        compute_breakdown = {}
        keys = list(state_dict.keys())
        
        for key in keys:
            param = state_dict[key]
            num_params = param.numel()
            
            # 识别层类型
            if 'conv' in key.lower() or 'conv2d' in key.lower():
                layer_type = 'conv2d'
            elif 'attention' in key.lower() or 'attn' in key.lower():
                layer_type = 'attention'
            elif 'embedding' in key.lower() or 'emb' in key.lower():
                layer_type = 'embedding'
            elif 'norm' in key.lower() or 'ln' in key.lower() or 'bn' in key.lower():
                layer_type = 'layernorm'
            elif 'weight' in key.lower() and len(param.shape) == 2:
                layer_type = 'linear'
            else:
                layer_type = 'linear'  # 默认
            
            # 计算该层的算力
            cost_per_param = self.OPERATION_COSTS.get(layer_type, 2.0)
            layer_compute = num_params * cost_per_param * self.batch_size
            
            if model_type == "transformer":
                # Transformer需要考虑序列长度
                layer_compute *= (self.sequence_length / 512.0)
            
            compute_breakdown[key] = layer_compute
        
        return compute_breakdown
    
    def _estimate_activation_memory(self, state_dict: Dict[str, torch.Tensor], 
                                    model_type: str) -> float:
        """估算激活值内存需求（GB）"""
        # 粗略估算：基于参数数量和模型类型
        total_params = sum(p.numel() for p in state_dict.values())
        
        if model_type == "transformer":
            # Transformer激活值较多（attention maps等）
            activation_ratio = 0.1  # 激活值约为参数的10%
        elif model_type == "cnn":
            # CNN激活值主要在特征图
            activation_ratio = 0.05  # 激活值约为参数的5%
        else:
            activation_ratio = 0.03  # 其他模型
        
        activation_params = total_params * activation_ratio
        activation_memory = activation_params * 4 / (1024**3)  # float32, 转换为GB
        
        return activation_memory * self.batch_size


def estimate_model_compute(model_path: str = None,
                          state_dict: Dict[str, torch.Tensor] = None,
                          model: nn.Module = None,
                          batch_size: int = 1,
                          model_type: str = "auto") -> Dict[str, float]:
    """
    便捷函数：估算模型算力需求
    
    Args:
        model_path: 模型文件路径（.pth或.onnx）
        state_dict: 模型参数字典
        model: PyTorch模型对象
        batch_size: 批次大小
        model_type: 模型类型
    
    Returns:
        估算结果
    """
    estimator = ComputeEstimator(batch_size=batch_size)
    
    if model is not None:
        state_dict = model.state_dict()
        return estimator.estimate_from_state_dict(state_dict, model_type)
    elif state_dict is not None:
        return estimator.estimate_from_state_dict(state_dict, model_type)
    elif model_path:
        if model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location='cpu')
            return estimator.estimate_from_state_dict(state_dict, model_type)
        else:
            raise ValueError("目前仅支持.pth格式，ONNX格式需要先转换")
    else:
        raise ValueError("必须提供model_path、state_dict或model之一")
