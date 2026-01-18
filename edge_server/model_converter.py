"""
模型转换模块
ONNX → PyTorch转换
读取state_dict文件
"""
import torch
import onnx
from typing import Dict, Optional, Any
from pathlib import Path


class ModelConverter:
    """模型转换器"""
    
    def onnx_to_pytorch(self, onnx_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        ONNX → PyTorch转换
        
        Args:
            onnx_path: ONNX模型路径
            output_path: 输出PyTorch模型路径（可选）
        
        Returns:
            转换结果字典，包含state_dict路径
        """
        try:
            import onnxruntime as ort
            from onnxruntime.tools.optimization_guide import optimize_model
            
            print(f"加载ONNX模型: {onnx_path}")
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 获取模型输入输出信息
            input_shapes = {}
            output_names = []
            
            for input_tensor in onnx_model.graph.input:
                shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append(1)  # 未知维度设为1
                input_shapes[input_tensor.name] = shape
            
            for output_tensor in onnx_model.graph.output:
                output_names.append(output_tensor.name)
            
            # 从ONNX提取参数
            state_dict = {}
            for initializer in onnx_model.graph.initializer:
                # 转换为PyTorch tensor
                tensor_data = onnx.numpy_helper.to_array(initializer)
                tensor = torch.from_numpy(tensor_data)
                state_dict[initializer.name] = tensor
            
            # 保存state_dict
            if output_path is None:
                output_path = str(Path(onnx_path).with_suffix('.pth'))
            
            torch.save(state_dict, output_path)
            print(f"ONNX模型已转换为PyTorch: {output_path}")
            
            return {
                'success': True,
                'state_dict_path': output_path,
                'input_shapes': input_shapes,
                'output_names': output_names,
                'metadata': {
                    'source_format': 'onnx',
                    'target_format': 'pytorch',
                    'onnx_path': onnx_path
                }
            }
        
        except Exception as e:
            print(f"ONNX转换失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def read_state_dict(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        读取state_dict文件
        
        Args:
            model_path: 模型文件路径（.pth）
        
        Returns:
            state_dict字典
        """
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            return state_dict
        except Exception as e:
            raise RuntimeError(f"读取state_dict失败: {str(e)}")
    
    def validate_model(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        验证模型state_dict
        
        Args:
            state_dict: 模型参数字典
        
        Returns:
            验证结果
        """
        total_params = sum(p.numel() for p in state_dict.values())
        total_size = sum(p.numel() * p.element_size() for p in state_dict.values())
        
        return {
            'total_params': total_params,
            'total_size_mb': total_size / (1024 * 1024),
            'num_layers': len(state_dict),
            'valid': True
        }
