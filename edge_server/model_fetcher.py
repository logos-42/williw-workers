"""
模型获取模块
从Hugging Face Hub或本地模型仓库获取模型
"""
import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path
import shutil


class ModelFetcher:
    """模型获取器"""
    
    def __init__(self, 
                 local_repo_path: str = "./model_repository",
                 huggingface_cache_dir: Optional[str] = None):
        """
        初始化模型获取器
        
        Args:
            local_repo_path: 本地模型仓库路径
            huggingface_cache_dir: Hugging Face缓存目录
        """
        self.local_repo_path = Path(local_repo_path)
        self.local_repo_path.mkdir(parents=True, exist_ok=True)
        
        self.hf_cache_dir = huggingface_cache_dir
        if self.hf_cache_dir:
            os.environ['HF_HOME'] = self.hf_cache_dir
    
    def fetch_model(self, 
                   model_name: str,
                   source: str = "huggingface",
                   model_format: str = "pytorch") -> Dict[str, Any]:
        """
        获取模型
        
        Args:
            model_name: 模型名称（Hugging Face model_id或本地模型名称）
            source: 来源（"huggingface"或"local"）
            model_format: 模型格式（"pytorch", "onnx"）
        
        Returns:
            模型信息字典，包含：
            - model_path: 模型文件路径
            - format: 模型格式
            - state_dict_path: state_dict路径（如果是PyTorch）
            - metadata: 模型元数据
        """
        if source == "huggingface":
            return self._fetch_from_huggingface(model_name, model_format)
        elif source == "local":
            return self._fetch_from_local(model_name, model_format)
        else:
            raise ValueError(f"不支持的模型来源: {source}")
    
    def _fetch_from_huggingface(self, model_id: str, model_format: str) -> Dict[str, Any]:
        """从Hugging Face Hub获取模型"""
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            from transformers import AutoModel, AutoConfig
            
            print(f"从Hugging Face下载模型: {model_id}")
            
            # 下载模型文件
            cache_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=self.hf_cache_dir,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.safetensors.index.json"]
            )
            
            # 尝试加载模型配置
            try:
                config = AutoConfig.from_pretrained(model_id, cache_dir=self.hf_cache_dir)
                model_type = config.model_type if hasattr(config, 'model_type') else 'unknown'
            except:
                model_type = 'unknown'
            
            # 查找模型文件
            model_path = None
            state_dict_path = None
            
            if model_format == "pytorch":
                # 查找pytorch_model.bin或model.safetensors
                pytorch_files = list(Path(cache_dir).rglob("pytorch_model.bin"))
                if pytorch_files:
                    model_path = str(pytorch_files[0])
                    state_dict_path = model_path
                else:
                    # 尝试加载为safetensors
                    safetensors_files = list(Path(cache_dir).rglob("model.safetensors"))
                    if safetensors_files:
                        # 加载并转换为pytorch
                        from safetensors.torch import load_file
                        state_dict = load_file(str(safetensors_files[0]))
                        # 保存为pytorch格式
                        state_dict_path = str(self.local_repo_path / f"{model_id.replace('/', '_')}.pth")
                        torch.save(state_dict, state_dict_path)
                        model_path = state_dict_path
                    else:
                        # 直接加载模型对象并保存state_dict
                        print(f"未找到预训练的pytorch模型文件，尝试加载模型对象...")
                        model = AutoModel.from_pretrained(model_id, cache_dir=self.hf_cache_dir)
                        state_dict_path = str(self.local_repo_path / f"{model_id.replace('/', '_')}.pth")
                        torch.save(model.state_dict(), state_dict_path)
                        model_path = state_dict_path
            
            elif model_format == "onnx":
                onnx_files = list(Path(cache_dir).rglob("*.onnx"))
                if onnx_files:
                    model_path = str(onnx_files[0])
                else:
                    raise FileNotFoundError(f"未找到ONNX模型文件: {model_id}")
            
            return {
                'model_path': model_path,
                'format': model_format,
                'state_dict_path': state_dict_path,
                'cache_dir': cache_dir,
                'metadata': {
                    'model_id': model_id,
                    'model_type': model_type,
                    'source': 'huggingface'
                }
            }
        
        except Exception as e:
            raise RuntimeError(f"从Hugging Face获取模型失败: {str(e)}")
    
    def _fetch_from_local(self, model_name: str, model_format: str) -> Dict[str, Any]:
        """从本地模型仓库获取模型"""
        model_dir = self.local_repo_path / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"本地模型不存在: {model_name}")
        
        # 查找模型文件
        model_path = None
        state_dict_path = None
        
        if model_format == "pytorch":
            # 查找.pth文件
            pth_files = list(model_dir.glob("*.pth"))
            if pth_files:
                model_path = str(pth_files[0])
                state_dict_path = model_path
            else:
                raise FileNotFoundError(f"未找到PyTorch模型文件: {model_name}")
        
        elif model_format == "onnx":
            onnx_files = list(model_dir.glob("*.onnx"))
            if onnx_files:
                model_path = str(onnx_files[0])
            else:
                raise FileNotFoundError(f"未找到ONNX模型文件: {model_name}")
        
        # 读取元数据（如果存在）
        metadata_path = model_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            'model_path': model_path,
            'format': model_format,
            'state_dict_path': state_dict_path,
            'metadata': {
                'model_name': model_name,
                'source': 'local',
                **metadata
            }
        }
    
    def list_local_models(self) -> list:
        """列出本地模型仓库中的所有模型"""
        models = []
        if self.local_repo_path.exists():
            for item in self.local_repo_path.iterdir():
                if item.is_dir():
                    models.append(item.name)
        return models
