"""
完整资源分配模块
使用LCR-AGA（遗传算法）和NL-APSO（粒子群优化）进行资源分配
"""
import sys
import os
sys.path.insert(0, '/work/lkc/youhua')

from typing import Dict, List, Any, Callable, Tuple
import numpy as np
from nl_apso import NLAPSO


class ResourceAllocator:
    """完整资源分配器 - 使用LCR-AGA和NL-APSO"""
    
    def __init__(self, num_nodes: int):
        """
        初始化资源分配器
        
        Args:
            num_nodes: 节点数量
        """
        self.num_nodes = num_nodes
        self.nl_apso = NLAPSO(
            num_nodes=num_nodes,
            population_size=200,
            max_iterations=300
        )
    
    def allocate_resources(self,
                          nodes: List,
                          compute_requirement: Dict[str, Any],
                          node_selection: np.ndarray) -> Dict[str, Any]:
        """
        分配资源到选中的节点
        
        Args:
            nodes: 节点列表
            compute_requirement: 算力需求
            node_selection: 节点选择方案（0或1数组）
        
        Returns:
            资源分配结果
        """
        print(f"\n=== 资源分配开始 ===")
        
        selected_nodes = [i for i in range(self.num_nodes) if node_selection[i] == 1]
        print(f"选中节点索引: {selected_nodes}")
        
        # 构建适应度函数
        fitness_func = self._build_fitness_function(nodes, compute_requirement, node_selection)
        
        # 构建约束函数
        constraints = self._build_constraints(nodes, compute_requirement, node_selection)
        
        # 使用NL-APSO进行优化
        print("开始NL-APSO优化...")
        best_position, fitness_history = self.nl_apso.optimize(
            fitness_func=fitness_func,
            constraints=constraints
        )
        
        # 解析最优位置
        node_selection_result, resource_targets = self.nl_apso.position_to_node_selection(best_position)
        
        print(f"最优适应度: {fitness_history[-1]:.4f}")
        print("=== 资源分配完成 ===\n")
        
        return {
            'node_selection': node_selection_result,
            'resource_targets': resource_targets,
            'fitness_history': fitness_history,
            'best_fitness': fitness_history[-1] if fitness_history else 0.0
        }
    
    def _build_fitness_function(self, 
                                nodes: List,
                                compute_requirement: Dict[str, Any],
                                node_selection: np.ndarray) -> Callable[[np.ndarray], float]:
        """
        构建适应度函数
        目标是最大化资源利用效率，最小化延迟和能耗
        """
        required_compute = compute_requirement.get('total_compute', 0.0)
        
        def fitness(position: np.ndarray) -> float:
            # 解析位置向量
            selection, resource_targets = self.nl_apso.position_to_node_selection(position)
            
            # 计算适应度（综合多个目标）
            total_compute = 0.0
            total_delay = 0.0
            total_energy = 0.0
            
            for i in range(self.num_nodes):
                if selection[i] == 1 and i < len(nodes):
                    node = nodes[i]
                    
                    # 算力贡献
                    compute_power = self._estimate_compute_power(node)
                    total_compute += compute_power * resource_targets[i].get('P_t', 0.5)
                    
                    # 延迟（基于网络）
                    network_delay = getattr(node, 'end_to_end_delay', 50.0)
                    total_delay += network_delay * resource_targets[i].get('delay', 0.5)
                    
                    # 能耗（基于电池）
                    battery_level = getattr(node, 'battery_level', 100.0)
                    energy_cost = (100.0 - battery_level) / 100.0
                    total_energy += energy_cost * resource_targets[i].get('P_ene', 0.5)
            
            # 适应度 = 算力收益 - 延迟惩罚 - 能耗惩罚
            compute_score = total_compute / (required_compute + 1e-6)
            delay_penalty = total_delay / 1000.0  # 归一化
            energy_penalty = total_energy / len(selected_nodes) if len(selected_nodes) > 0 else 0
            
            fitness_value = compute_score * 100.0 - delay_penalty * 10.0 - energy_penalty * 5.0
            
            return fitness_value
        
        return fitness
    
    def _build_constraints(self,
                          nodes: List,
                          compute_requirement: Dict[str, Any],
                          node_selection: np.ndarray) -> List[Callable]:
        """
        构建约束函数
        约束包括：算力约束、内存约束、GPU约束等
        """
        required_compute = compute_requirement.get('total_compute', 0.0)
        required_memory = compute_requirement.get('memory_required', 0.0)
        required_gpu = compute_requirement.get('gpu_required', False)
        
        constraints = []
        
        # 约束1: 算力约束
        def compute_constraint(position: np.ndarray) -> float:
            selection, resource_targets = self.nl_apso.position_to_node_selection(position)
            total_compute = 0.0
            
            for i in range(self.num_nodes):
                if selection[i] == 1 and i < len(nodes):
                    compute_power = self._estimate_compute_power(nodes[i])
                    total_compute += compute_power * resource_targets[i].get('P_t', 0.5)
            
            # g(x) = required - total (违反时 > 0)
            return required_compute - total_compute
        
        constraints.append(compute_constraint)
        
        # 约束2: 内存约束
        def memory_constraint(position: np.ndarray) -> float:
            selection, resource_targets = self.nl_apso.position_to_node_selection(position)
            total_memory = 0.0
            
            for i in range(self.num_nodes):
                if selection[i] == 1 and i < len(nodes):
                    node = nodes[i]
                    max_memory = getattr(node, 'max_memory_mb', 8192) / 1024.0  # GB
                    memory_usage = getattr(node, 'memory_usage', 0.0) / 100.0
                    available_memory = max_memory * (1 - memory_usage)
                    total_memory += available_memory * resource_targets[i].get('P_mem', 0.5)
            
            return required_memory - total_memory
        
        constraints.append(memory_constraint)
        
        # 约束3: GPU约束（如果需要GPU）
        if required_gpu:
            def gpu_constraint(position: np.ndarray) -> float:
                selection, _ = self.nl_apso.position_to_node_selection(position)
                gpu_count = 0
                
                for i in range(self.num_nodes):
                    if selection[i] == 1 and i < len(nodes):
                        if getattr(nodes[i], 'gpu_available', False):
                            gpu_count += 1
                
                # 至少需要1个GPU节点
                return 1.0 - gpu_count
            
            constraints.append(gpu_constraint)
        
        return constraints
    
    def _estimate_compute_power(self, node) -> float:
        """估算节点算力"""
        if not getattr(node, 'gpu_available', False):
            cpu_cores = getattr(node, 'cpu_cores', 4)
            return cpu_cores * 10.0
        
        gpu_name = getattr(node, 'gpu_name', '').lower()
        gpu_usage = getattr(node, 'gpu_usage_percent', 0.0)
        
        gpu_compute_map = {
            'rtx 4090': 80000.0,
            'rtx 4080': 50000.0,
            'rtx 3090': 36000.0,
            'rtx 3080': 30000.0,
            'a100': 312000.0,
            'v100': 125000.0,
            't4': 8000.0,
        }
        
        base_compute = 5000.0
        for gpu_key, compute in gpu_compute_map.items():
            if gpu_key in gpu_name:
                base_compute = compute
                break
        
        return base_compute * (1 - gpu_usage / 100.0)
