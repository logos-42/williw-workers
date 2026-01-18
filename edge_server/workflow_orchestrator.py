"""
工作流编排器
编排完整的推理流程：模型获取→转换→算力估算→节点选择→资源分配→模型切分→推理→结果集成
"""
import sys
import os
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

# 添加lkc项目路径
sys.path.insert(0, '/work/lkc/youhua')

from edge_server.model_fetcher import ModelFetcher
from edge_server.model_converter import ModelConverter
from edge_server.compute_estimator import ComputeEstimator
from interface_layer.node_info_api import NodeInfoAPI
from models.inference_engine import DistributedInferenceEngine
from models.result_merger import ResultMerger

# 导入完整算法层
try:
    from algorithms.task_scheduler import TaskScheduler
    FULL_ALGORITHMS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 完整算法层不可用: {e}")
    FULL_ALGORITHMS_AVAILABLE = False


class WorkflowOrchestrator:
    """工作流编排器"""
    
    def __init__(self):
        """初始化工作流编排器"""
        self.model_fetcher = ModelFetcher()
        self.model_converter = ModelConverter()
        self.compute_estimator = ComputeEstimator()
        self.node_info_api = NodeInfoAPI()
        self.inference_engine = DistributedInferenceEngine()
        self.result_merger = ResultMerger()
        
        # 初始化完整算法层
        self.task_scheduler = None
        if FULL_ALGORITHMS_AVAILABLE:
            self.task_scheduler = TaskScheduler()
            print("✓ 完整算法层已加载")
        else:
            print("警告: 使用简化算法层")
    
    def execute_inference_workflow(self,
                                   model_name: str,
                                   model_source: str,
                                   input_data: Dict[str, Any],
                                   parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行完整的推理工作流
        
        Args:
            model_name: 模型名称
            model_source: 模型来源（"huggingface"或"local"）
            input_data: 输入数据
            parameters: 推理参数（batch_size等）
        
        Returns:
            推理结果字典
        """
        if parameters is None:
            parameters = {}
        
        try:
            print(f"\n{'='*70}")
            print("开始执行推理工作流")
            print(f"{'='*70}\n")
            
            # 步骤1: 获取模型
            print("步骤1: 获取模型...")
            model_info = self.model_fetcher.fetch_model(
                model_name=model_name,
                source=model_source,
                model_format="pytorch"
            )
            print(f"✓ 模型获取成功: {model_info['model_path']}")
            
            # 步骤2: 转换模型（如果需要）
            if model_info['format'] == 'onnx':
                print("\n步骤2: 转换ONNX模型为PyTorch...")
                conversion_result = self.model_converter.onnx_to_pytorch(
                    model_info['model_path']
                )
                if conversion_result['success']:
                    model_info['state_dict_path'] = conversion_result['state_dict_path']
                    print(f"✓ 模型转换成功: {model_info['state_dict_path']}")
                else:
                    raise RuntimeError(f"模型转换失败: {conversion_result.get('error')}")
            else:
                print("\n步骤2: 跳过转换（已经是PyTorch格式）")
            
            # 步骤3: 读取state_dict
            print("\n步骤3: 读取state_dict...")
            state_dict = self.model_converter.read_state_dict(model_info['state_dict_path'])
            validation = self.model_converter.validate_model(state_dict)
            print(f"✓ state_dict读取成功: {validation['total_params']} 参数, {validation['total_size_mb']:.2f} MB")
            
            # 步骤4: 估算算力需求
            print("\n步骤4: 估算模型算力需求（保守估算）...")
            batch_size = parameters.get('batch_size', 1)
            compute_req = self.compute_estimator.estimate_from_state_dict(
                state_dict,
                batch_size=batch_size
            )
            print(f"✓ 算力估算完成:")
            print(f"  - 总算力需求: {compute_req['total_compute']:.2f} GFLOPS")
            print(f"  - 内存需求: {compute_req['memory_required']:.2f} GB")
            print(f"  - 需要GPU: {compute_req['gpu_required']}")
            print(f"  - 估算延迟: {compute_req['estimated_latency']:.2f} ms")
            
            # 步骤5: 获取节点信息
            print("\n步骤5: 获取可用节点信息...")
            available_nodes = self.node_info_api.get_available_nodes()
            print(f"✓ 获取到 {len(available_nodes)} 个可用节点")
            
            # 步骤6: 调用完整算法层（节点选择、路径优化、资源分配、模型切分、任务分发）
            print("\n步骤6: 调用完整算法层进行节点选择、资源分配和模型切分...")
            
            if self.task_scheduler:
                # 使用完整算法层
                algorithm_result = self.task_scheduler.process_inference_task(
                    compute_requirement=compute_req,
                    available_nodes=available_nodes,
                    state_dict=state_dict,
                    input_data=input_data
                )
                
                if not algorithm_result['success']:
                    raise RuntimeError(f"算法层处理失败: {algorithm_result.get('error', '未知错误')}")
                
                # 提取结果
                selected_node_ids = algorithm_result['node_selection']['primary_nodes']
                # 根据node_id找到对应的节点对象
                selected_nodes = [n for n in available_nodes if getattr(n, 'node_id', '') in selected_node_ids]
                model_shards = algorithm_result['model_shards']
                distribution_result = algorithm_result['distribution']
                
                print(f"✓ 完整算法层处理完成")
                print(f"  - 选中主节点数: {len(selected_nodes)}")
                print(f"  - 备份节点数: {len(algorithm_result['node_selection']['backup_nodes'])}")
                print(f"  - 模型分片数: {len(model_shards)}")
                print(f"  - 路径优化完成: {len(algorithm_result['routing']['routing_table'])} 条路径")
                print(f"  - 资源分配完成: 适应度 {algorithm_result['resource_allocation']['best_fitness']:.4f}")
                
                # Megaphone Mode分发已经完成，直接使用分发结果
                inference_result = {
                    'node_results': distribution_result['node_results'],
                    'total_time': distribution_result['total_time'],
                    'distribution_mode': 'megaphone'
                }
            else:
                # 回退到简化版本
                allocation_result = self._allocate_resources_and_split_model(
                    compute_req=compute_req,
                    available_nodes=available_nodes,
                    state_dict=state_dict,
                    model_info=model_info
                )
                selected_nodes = [n for n in available_nodes if getattr(n, 'node_id', '') in allocation_result['selected_nodes']]
                model_shards = allocation_result['model_shards']
                
                print(f"✓ 简化算法层处理完成")
                print(f"  - 选中节点数: {len(selected_nodes)}")
                print(f"  - 模型分片数: {len(model_shards)}")
                
                # 步骤7: 分布式推理（简化版本）
                print("\n步骤7: 执行分布式推理...")
                inference_result = self.inference_engine.infer(
                    model_shards=model_shards,
                    input_data=input_data,
                    nodes=selected_nodes,
                    parameters=parameters
                )
                print(f"✓ 分布式推理完成，耗时: {inference_result['total_time']:.2f} ms")
            
            # 步骤8: 集成结果
            print("\n步骤8: 集成推理结果...")
            final_result = self.result_merger.merge(inference_result)
            print(f"✓ 结果集成完成")
            
            print(f"\n{'='*70}")
            print("推理工作流执行完成！")
            print(f"{'='*70}\n")
            
            # 构建返回结果
            result = {
                'status': 'success',
                'result': final_result,
                'inference_time': inference_result['total_time'],
                'compute_requirement': compute_req,
                'model_info': model_info
            }
            
            if self.task_scheduler:
                # 完整算法层的结果
                result['nodes_used'] = selected_node_ids
                result['algorithm_details'] = {
                    'node_selection': algorithm_result['node_selection'],
                    'routing': algorithm_result['routing'],
                    'resource_allocation': algorithm_result['resource_allocation'],
                    'distribution_mode': 'megaphone'
                }
            else:
                result['nodes_used'] = allocation_result['selected_nodes']
            
            return result
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _allocate_resources_and_split_model(self,
                                           compute_req: Dict[str, Any],
                                           available_nodes: List,
                                           state_dict: Dict,
                                           model_info: Dict) -> Dict[str, Any]:
        """
        调用算法层进行节点选择、路径优化、资源分配和模型切分
        
        Args:
            compute_req: 算力需求
            available_nodes: 可用节点列表
            state_dict: 模型state_dict
            model_info: 模型信息
        
        Returns:
            分配结果，包含选中的节点和模型分片
        """
        # 简化的节点选择（如果lkc不可用）
        if not LKC_AVAILABLE or len(available_nodes) == 0:
            # 使用简化选择逻辑
            selected_nodes = self._simple_node_selection(available_nodes, compute_req)
            model_shards = self._simple_model_split(state_dict, selected_nodes)
            return {
                'selected_nodes': [n.node_id for n in selected_nodes],
                'model_shards': model_shards,
                'resource_allocation': {}
            }
        
        # 使用lkc算法层（需要先创建TaskScheduler等，这里简化处理）
        # 实际应该调用lkc.core.scheduler.TaskScheduler
        selected_nodes = self._simple_node_selection(available_nodes, compute_req)
        model_shards = self._simple_model_split(state_dict, selected_nodes)
        
        # 路径优化（使用D-CACO）
        if self.dcaco and len(selected_nodes) > 1:
            # 更新D-CACO的节点数量
            self.dcaco.num_nodes = len(selected_nodes)
            # 执行路径优化（这里简化，实际需要传入节点间的距离矩阵）
            # best_path = self.dcaco.optimize(...)
        
        return {
            'selected_nodes': [n.node_id for n in selected_nodes],
            'model_shards': model_shards,
            'resource_allocation': {
                'method': 'simplified',
                'note': '使用简化算法（lkc完整算法需要恢复）'
            }
        }
    
    def _simple_node_selection(self, available_nodes: List, compute_req: Dict[str, Any]) -> List:
        """简化的节点选择逻辑"""
        selected = []
        required_compute = compute_req['total_compute']
        required_memory = compute_req['memory_required']
        required_gpu = compute_req['gpu_required']
        
        # 过滤节点
        candidates = []
        for node in available_nodes:
            # 检查GPU需求
            if required_gpu and not node.gpu_available:
                continue
            
            # 检查算力
            available_compute = node.compute_power * (1 - node.cpu_usage / 100.0)
            if available_compute < required_compute * 0.3:  # 至少需要30%的算力
                continue
            
            # 检查内存
            available_memory = 8.0 * (1 - node.memory_usage / 100.0)  # 假设8GB总内存
            if available_memory < required_memory * 0.8:
                continue
            
            # 检查在线和空闲状态
            if not node.is_online or not node.is_idle:
                continue
            
            candidates.append(node)
        
        # 按算力排序，选择前几个
        candidates.sort(key=lambda n: n.compute_power * (1 - n.cpu_usage / 100.0), reverse=True)
        
        # 选择足够的节点（至少2个，最多5个）
        num_nodes = min(max(2, int(required_compute / 100.0)), 5, len(candidates))
        selected = candidates[:num_nodes]
        
        return selected
    
    def _simple_model_split(self, state_dict: Dict, nodes: List) -> List[Dict[str, Any]]:
        """简化的模型切分逻辑"""
        layer_names = list(state_dict.keys())
        total_layers = len(layer_names)
        
        if total_layers == 0:
            return []
        
        num_nodes = len(nodes)
        layers_per_node = total_layers // num_nodes
        
        shards = []
        output_dir = Path("./model_shards")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, node in enumerate(nodes):
            start_idx = i * layers_per_node
            if i == num_nodes - 1:
                # 最后一个节点包含剩余的所有层
                end_idx = total_layers
            else:
                end_idx = (i + 1) * layers_per_node
            
            node_layers = layer_names[start_idx:end_idx]
            shard_state_dict = {name: state_dict[name] for name in node_layers}
            
            shard_path = output_dir / f"shard_{node.node_id}.pth"
            import torch
            torch.save(shard_state_dict, shard_path)
            
            shards.append({
                'node_id': node.node_id,
                'shard_path': str(shard_path),
                'layer_names': node_layers,
                'layer_indices': list(range(start_idx, end_idx)),
                'layer_count': len(node_layers)
            })
        
        return shards
