"""
完整任务调度器
协调节点选择、资源分配、路径优化、模型切分和任务分发
"""
from typing import Dict, List, Any, Optional
import numpy as np

from .node_selection import NodeSelector
from .resource_allocator import ResourceAllocator
from .path_optimizer import PathOptimizer
from .model_splitter import ModelSplitter
from .task_distributor import TaskDistributor


class TaskScheduler:
    """完整任务调度器 - 协调所有算法"""
    
    def __init__(self):
        """初始化任务调度器"""
        self.node_selector = NodeSelector()
        self.resource_allocator = None  # 将在process_task时初始化
        self.path_optimizer = None  # 将在process_task时初始化
        self.model_splitter = ModelSplitter()
        self.task_distributor = TaskDistributor()
    
    def process_inference_task(self,
                              compute_requirement: Dict[str, Any],
                              available_nodes: List,
                              state_dict: Dict[str, Any],
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理推理任务 - 完整流程
        
        Args:
            compute_requirement: 算力需求
            available_nodes: 可用节点列表
            state_dict: 模型state_dict
            input_data: 输入数据
        
        Returns:
            完整的处理结果
        """
        print(f"\n{'='*70}")
        print("=== 任务调度器开始处理推理任务 ===")
        print(f"{'='*70}\n")
        
        # 步骤1: 节点选择
        print("阶段1: 节点选择")
        selection_result = self.node_selector.select_nodes(
            available_nodes=available_nodes,
            compute_requirement=compute_requirement
        )
        
        if not selection_result['success']:
            return {
                'success': False,
                'error': selection_result.get('message', '节点选择失败'),
                'stage': 'node_selection'
            }
        
        primary_nodes = selection_result['primary_nodes']
        backup_nodes = selection_result['backup_nodes']
        
        print(f"✓ 选择了 {len(primary_nodes)} 个主节点和 {len(backup_nodes)} 个备份节点\n")
        
        # 步骤2: 路径优化（D-CACO）
        print("阶段2: 路径优化（D-CACO）")
        self.path_optimizer = PathOptimizer(num_nodes=len(primary_nodes))
        routing_result = self.path_optimizer.optimize_routing(primary_nodes)
        routing_table = routing_result.get('routing_table', {})
        print(f"✓ 路径优化完成，优化了 {len(routing_table)} 条路径\n")
        
        # 步骤3: 资源分配（NL-APSO）
        print("阶段3: 资源分配（NL-APSO）")
        # 构建节点选择数组
        all_nodes = primary_nodes + backup_nodes
        node_selection = np.zeros(len(all_nodes))
        node_selection[:len(primary_nodes)] = 1  # 主节点标记为1
        
        self.resource_allocator = ResourceAllocator(num_nodes=len(all_nodes))
        allocation_result = self.resource_allocator.allocate_resources(
            nodes=all_nodes,
            compute_requirement=compute_requirement,
            node_selection=node_selection
        )
        print(f"✓ 资源分配完成，最优适应度: {allocation_result['best_fitness']:.4f}\n")
        
        # 步骤4: 模型切分
        print("阶段4: 模型切分")
        model_shards = self.model_splitter.split_by_layers(
            state_dict=state_dict,
            nodes=primary_nodes,
            split_strategy='compute'  # 按算力切分
        )
        print(f"✓ 模型切分完成，共 {len(model_shards)} 个分片\n")
        
        # 步骤5: 任务分发（Megaphone Mode）
        print("阶段5: 任务分发（Megaphone Mode）")
        distribution_result = self.task_distributor.distribute_with_megaphone_mode(
            model_shards=model_shards,
            nodes=primary_nodes,
            input_data=input_data,
            routing_table=routing_table
        )
        print(f"✓ 任务分发完成\n")
        
        print(f"{'='*70}")
        print("=== 任务调度完成 ===")
        print(f"{'='*70}\n")
        
        return {
            'success': True,
            'node_selection': {
                'primary_nodes': [getattr(n, 'node_id', 'unknown') for n in primary_nodes],
                'backup_nodes': [getattr(n, 'node_id', 'unknown') for n in backup_nodes],
                'total_compute': selection_result['total_compute']
            },
            'routing': {
                'routing_table': routing_result.get('routing_table', {}),
                'performance': routing_result.get('performance', {})
            },
            'resource_allocation': {
                'node_selection': allocation_result['node_selection'].tolist(),
                'best_fitness': allocation_result['best_fitness']
            },
            'model_shards': model_shards,
            'distribution': distribution_result
        }
