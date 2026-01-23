"""
用户节点端完整流程
1. 发送模型名称给 Worker
2. 从 Hugging Face 下载模型
3. 提取 state_dict 并生成元数据
4. 上传元数据到 Hugging Face 公共仓库
5. 接收 Worker 的分配方案
6. 根据方案切分和分发模型
"""
import os
import json
import requests
import torch
from transformers import AutoModel
from huggingface_hub import HfApi, login
from pathlib import Path
from typing import Dict, List, Any, Optional
import time


class ComputeEstimator:
    """算力估算器（用于生成元数据）"""
    
    OPERATION_COSTS = {
        'conv2d': 2.0,
        'linear': 2.0,
        'attention': 4.0,  # Attention 层算力需求更高
        'layernorm': 1.0,
        'embedding': 0.5,
        'activation': 0.1,
        'pooling': 0.2,
    }
    
    def __init__(self, batch_size: int = 1, sequence_length: int = 512):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
    
    def estimate_layer_compute(self, layer_name: str, tensor: torch.Tensor, 
                              model_type: str = "transformer") -> float:
        """估算单层算力需求（GFLOPS）"""
        num_params = tensor.numel()
        layer_type = self._identify_layer_type(layer_name, tensor)
        cost_per_param = self.OPERATION_COSTS.get(layer_type, 2.0)
        layer_compute = num_params * cost_per_param * self.batch_size
        
        if model_type == "transformer":
            layer_compute *= (self.sequence_length / 512.0)
        
        return layer_compute
    
    def _identify_layer_type(self, layer_name: str, tensor: torch.Tensor) -> str:
        """识别层类型"""
        name_lower = layer_name.lower()
        if 'conv' in name_lower or 'conv2d' in name_lower:
            return 'conv2d'
        elif 'attention' in name_lower or 'attn' in name_lower:
            return 'attention'
        elif 'embedding' in name_lower or 'emb' in name_lower:
            return 'embedding'
        elif 'norm' in name_lower or 'ln' in name_lower or 'bn' in name_lower:
            return 'layernorm'
        elif 'weight' in name_lower and len(tensor.shape) == 2:
            return 'linear'
        else:
            return 'linear'


class NodeClient:
    """用户节点客户端 - 完整流程"""
    
    def __init__(self, 
                 worker_url: str,
                 hf_token: Optional[str] = None,
                 metadata_repo: str = "logos42/williw",
                 node_id: str = "node_001"):
        """
        初始化节点客户端
        
        Args:
            worker_url: Worker API 地址
            hf_token: Hugging Face Token（可选，从环境变量读取）
            metadata_repo: 元数据仓库名称
            node_id: 节点ID
        """
        self.worker_url = worker_url.rstrip('/')
        self.metadata_repo = metadata_repo
        self.node_id = node_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        if self.hf_token:
            login(token=self.hf_token)
            self.hf_api = HfApi(token=self.hf_token)
        else:
            self.hf_api = HfApi()
    
    def execute_complete_workflow(self, 
                                  model_name: str,
                                  batch_size: int = 1,
                                  sequence_length: int = 512) -> Dict[str, Any]:
        """
        执行完整流程
        
        Args:
            model_name: 模型名称（如 "meta-llama/Llama-3.2-1B-Instruct"）
            batch_size: 批次大小
            sequence_length: 序列长度
        
        Returns:
            完整流程结果
        """
        print(f"\n{'='*70}")
        print(f"=== 用户节点完整流程开始 ===")
        print(f"模型: {model_name}")
        print(f"节点ID: {self.node_id}")
        print(f"{'='*70}\n")
        
        workflow_result = {
            "model_name": model_name,
            "node_id": self.node_id,
            "steps": {}
        }
        
        try:
            # 步骤1: 发送模型名称给 Worker
            print("步骤1: 发送模型名称给 Worker...")
            request_result = self._send_model_request(model_name)
            workflow_result["steps"]["request"] = request_result
            print(f"✓ Worker 已接收请求\n")
            
            # 步骤2: 从 Hugging Face 下载模型
            print("步骤2: 从 Hugging Face 下载模型...")
            model_path = self._download_model(model_name)
            workflow_result["steps"]["download"] = {"model_path": model_path}
            print(f"✓ 模型下载完成: {model_path}\n")
            
            # 步骤3: 提取 state_dict 并生成元数据
            print("步骤3: 提取 state_dict 并生成元数据...")
            metadata = self._generate_metadata(model_name, model_path, batch_size, sequence_length)
            workflow_result["steps"]["metadata"] = {"metadata_file": metadata["filename"]}
            print(f"✓ 元数据生成完成: {metadata['filename']}\n")
            
            # 步骤4: 上传元数据到 Hugging Face 公共仓库
            print("步骤4: 上传元数据到 Hugging Face 公共仓库...")
            upload_result = self._upload_metadata(metadata["filename"], metadata["data"])
            workflow_result["steps"]["upload"] = upload_result
            print(f"✓ 元数据已上传到: {self.metadata_repo}/{metadata['filename']}\n")
            
            # 步骤5: 通知 Worker 元数据已准备好，触发算法处理
            print("步骤5: 通知 Worker 元数据已准备好，触发算法处理...")
            process_result = self._trigger_worker_process(model_name)
            workflow_result["steps"]["process"] = process_result
            
            # 从 process_result 中直接获取分配方案
            if process_result.get("success"):
                split_plan = process_result
                print(f"✓ Worker 已生成分配方案\n")
            else:
                raise RuntimeError("Worker 处理失败")
            
            # 步骤6: 保存分配方案
            workflow_result["steps"]["split_plan"] = {
                "model_name": split_plan.get("model_name"),
                "split_plan": split_plan.get("split_plan"),
                "megaphone_plan": split_plan.get("megaphone_plan"),
                "selected_nodes": split_plan.get("selected_nodes")
            }
            
            # 步骤7: 根据方案切分和分发模型
            print("步骤7: 根据方案切分和分发模型...")
            # 使用保存的 split_plan
            split_plan_data = workflow_result["steps"]["split_plan"]
            split_result = self._execute_split(model_path, split_plan_data)
            workflow_result["steps"]["split"] = split_result
            print(f"✓ 模型切分完成\n")
            
            workflow_result["success"] = True
            print(f"\n{'='*70}")
            print("=== 完整流程执行成功 ===")
            print(f"{'='*70}\n")
            
        except Exception as e:
            workflow_result["success"] = False
            workflow_result["error"] = str(e)
            print(f"\n❌ 流程执行失败: {e}\n")
            import traceback
            traceback.print_exc()
        
        return workflow_result
    
    def _send_model_request(self, model_name: str) -> Dict[str, Any]:
        """发送模型请求给 Worker（只发送请求，不等待处理）"""
        url = f"{self.worker_url}/api/request"
        payload = {
            "model_name": model_name,
            "node_id": self.node_id,
            "timestamp": time.time()
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _trigger_worker_process(self, model_name: str) -> Dict[str, Any]:
        """通知 Worker 元数据已准备好，触发算法处理"""
        url = f"{self.worker_url}/api/process"
        payload = {
            "model_name": model_name,
            "node_id": self.node_id,
            "metadata_repo": self.metadata_repo,
            "timestamp": time.time()
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _download_model(self, model_name: str) -> str:
        """从 Hugging Face 下载模型"""
        cache_dir = Path("./models_cache") / model_name.replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  下载模型到: {cache_dir}")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=False
        )
        
        return str(cache_dir)
    
    def _generate_metadata(self, 
                          model_name: str,
                          model_path: str,
                          batch_size: int,
                          sequence_length: int) -> Dict[str, Any]:
        """提取 state_dict 并生成元数据"""
        # 加载模型
        model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
        state_dict = model.state_dict()
        
        # 检测模型类型
        model_type = "transformer" if any(
            "transformer" in k.lower() or "attention" in k.lower() 
            for k in state_dict.keys()
        ) else "mlp"
        
        # 初始化算力估算器
        estimator = ComputeEstimator(batch_size=batch_size, sequence_length=sequence_length)
        
        # 提取元数据
        metadata = {
            "model_name": model_name,
            "model_type": model_type,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "layers": []
        }
        
        total_compute = 0.0
        
        # 按顺序提取（确保和 state_dict.keys() 的顺序一致）
        for name, tensor in state_dict.items():
            layer_compute = estimator.estimate_layer_compute(name, tensor, model_type)
            total_compute += layer_compute
            
            metadata["layers"].append({
                "name": name,
                "shape": list(tensor.shape),
                "num_params": tensor.numel(),
                "compute_required": layer_compute,
                "layer_type": estimator._identify_layer_type(name, tensor),
                "dtype": str(tensor.dtype)
            })
        
        metadata["total_compute"] = total_compute
        metadata["total_layers"] = len(metadata["layers"])
        metadata["generated_at"] = time.time()
        metadata["node_id"] = self.node_id
        
        # 保存为 JSON
        filename = model_name.replace("/", "_") + "_metadata.json"
        filepath = Path("./metadata") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  元数据大小: {filepath.stat().st_size / 1024:.2f} KB")
        print(f"  总算力需求: {total_compute:.2f} GFLOPS")
        print(f"  总层数: {len(metadata['layers'])}")
        
        return {
            "filename": filename,
            "filepath": str(filepath),
            "data": metadata
        }
    
    def _upload_metadata(self, filename: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """上传元数据到 Hugging Face 公共仓库"""
        filepath = Path("./metadata") / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"元数据文件不存在: {filepath}")
        
        self.hf_api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filename,
            repo_id=self.metadata_repo,
            repo_type="model",
            token=self.hf_token
        )
        
        return {
            "repo": self.metadata_repo,
            "filename": filename,
            "url": f"https://huggingface.co/{self.metadata_repo}/resolve/main/{filename}"
        }
    
    def _get_split_plan(self, model_name: str) -> Dict[str, Any]:
        """从 Worker 获取分配方案（从 /api/process 的响应中获取）"""
        # 如果之前已经调用了 /api/process，可以直接使用返回的结果
        # 这里简化处理，实际可以保存 process_result
        url = f"{self.worker_url}/api/get-plan"
        payload = {
            "model_name": model_name,
            "node_id": self.node_id
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def _execute_split(self, 
                       model_path: str,
                       split_plan: Dict[str, Any]) -> Dict[str, Any]:
        """根据方案切分模型（按每层的算力切分）"""
        # 加载模型
        model_name = split_plan.get("model_name")
        model = AutoModel.from_pretrained(model_name, cache_dir=model_path)
        state_dict = model.state_dict()
        
        # 获取本节点的切分方案
        split_plan_data = split_plan.get("split_plan", {})
        my_plan = split_plan_data.get(self.node_id, {})
        my_layer_names = my_plan.get("layer_names", [])
        
        if not my_layer_names:
            raise ValueError(f"节点 {self.node_id} 没有分配到任何层")
        
        # 提取本节点的层
        my_shard = {name: state_dict[name].clone() for name in my_layer_names}
        
        # 保存分片
        output_dir = Path("./model_shards") / self.node_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        shard_path = output_dir / f"shard_{self.node_id}.pth"
        torch.save(my_shard, shard_path)
        
        total_params = sum(p.numel() for p in my_shard.values())
        shard_size_mb = sum(p.numel() * 4 for p in my_shard.values()) / (1024 * 1024)
        
        print(f"  节点 {self.node_id} 分片:")
        print(f"    层数: {len(my_layer_names)}")
        print(f"    参数: {total_params}")
        print(f"    大小: {shard_size_mb:.2f} MB")
        print(f"    路径: {shard_path}")
        print(f"    算力利用率: {my_plan.get('compute_utilization', 0):.2%}")
        
        return {
            "node_id": self.node_id,
            "shard_path": str(shard_path),
            "layer_names": my_layer_names,
            "total_params": total_params,
            "shard_size_mb": shard_size_mb,
            "compute_utilization": my_plan.get("compute_utilization", 0)
        }


# 使用示例
if __name__ == "__main__":
    import sys
    
    # 配置
    WORKER_URL = os.getenv("WORKER_URL", "https://your-worker.workers.dev")
    HF_TOKEN = os.getenv("HF_TOKEN")
    METADATA_REPO = os.getenv("METADATA_REPO", "logos42/williw")
    NODE_ID = os.getenv("NODE_ID", "node_001")
    
    # 模型名称
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-1B-Instruct"
    
    # 创建客户端并执行流程
    client = NodeClient(
        worker_url=WORKER_URL,
        hf_token=HF_TOKEN,
        metadata_repo=METADATA_REPO,
        node_id=NODE_ID
    )
    
    result = client.execute_complete_workflow(
        model_name=model_name,
        batch_size=1,
        sequence_length=512
    )
    
    print("\n=== 流程结果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
