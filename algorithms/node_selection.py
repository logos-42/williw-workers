"""
完整节点选择算法
根据算力需求、资源阈值、GPU要求等选择主节点和备份节点
"""
import sys
import os
sys.path.insert(0, '/work/lkc/youhua')

from typing import List, Dict, Any, Tuple
import numpy as np


class NodeSelector:
    """完整节点选择算法"""
    
    def __init__(self, 
                 resource_thresholds: Dict[str, float] = None,
                 min_backup_nodes: int = 2):
        """
        初始化节点选择器
        
        Args:
            resource_thresholds: 资源阈值配置
            min_backup_nodes: 最小备份节点数
        """
        self.resource_thresholds = resource_thresholds or {
            'gpu_usage_max': 80.0,      # GPU使用率上限（%）
            'cpu_usage_max': 85.0,      # CPU使用率上限（%）
            'memory_usage_max': 80.0,   # 内存使用率上限（%）
            'battery_level_min': 20.0,  # 最低电池电量（%）
            'bandwidth_min': 1.0        # 最低带宽（Mbps）
        }
        self.min_backup_nodes = min_backup_nodes
    
    def estimate_compute_power(self, node) -> float:
        """
        估算节点算力（基于GPU）
        
        Args:
            node: MobileNode对象
        
        Returns:
            算力值（GFLOPS）
        """
        if not hasattr(node, 'gpu_available') or not node.gpu_available:
            # CPU算力（保守估算）
            cpu_cores = getattr(node, 'cpu_cores', 4)
            return cpu_cores * 10.0  # 每核心约10 GFLOPS
        
        # GPU算力估算（基于GPU型号）
        gpu_name = getattr(node, 'gpu_name', '').lower()
        gpu_usage = getattr(node, 'gpu_usage_percent', 0.0)
        
        # GPU算力映射（TFLOPS转换为GFLOPS）
        gpu_compute_map = {
            'rtx 4090': 80000.0,      # ~80 TFLOPS
            'rtx 4080': 50000.0,      # ~50 TFLOPS
            'rtx 3090': 36000.0,      # ~36 TFLOPS
            'rtx 3080': 30000.0,      # ~30 TFLOPS
            'rtx 3070': 20000.0,      # ~20 TFLOPS
            'a100': 312000.0,         # ~312 TFLOPS
            'v100': 125000.0,         # ~125 TFLOPS
            't4': 8000.0,             # ~8 TFLOPS
            'k80': 5600.0,            # ~5.6 TFLOPS
        }
        
        base_compute = 5000.0  # 默认GPU算力（GFLOPS）
        for gpu_key, compute in gpu_compute_map.items():
            if gpu_key in gpu_name:
                base_compute = compute
                break
        
        # 考虑GPU使用率
        available_compute = base_compute * (1 - gpu_usage / 100.0)
        
        # 考虑计算能力指标
        compute_capability = getattr(node, 'compute_capability', 1.0)
        available_compute *= compute_capability
        
        return available_compute
    
    def check_resource_constraints(self, node, compute_requirement: float) -> Tuple[bool, List[str]]:
        """
        检查节点是否满足资源约束
        
        Args:
            node: MobileNode对象
            compute_requirement: 需要的算力（GFLOPS）
        
        Returns:
            (是否满足, 不满足的原因列表)
        """
        violations = []
        
        # 检查在线状态
        if not getattr(node, 'is_online', False):
            violations.append("节点离线")
            return False, violations
        
        # 检查空闲状态
        if not getattr(node, 'is_idle', True):
            violations.append("节点忙碌")
            return False, violations
        
        # 检查GPU需求
        if compute_requirement > 1000.0:  # 需要GPU的任务
            if not getattr(node, 'gpu_available', False):
                violations.append("无GPU")
                return False, violations
            
            gpu_usage = getattr(node, 'gpu_usage_percent', 0.0)
            if gpu_usage > self.resource_thresholds['gpu_usage_max']:
                violations.append(f"GPU使用率过高({gpu_usage:.1f}%)")
                return False, violations
        
        # 检查CPU使用率
        cpu_usage = getattr(node, 'cpu_usage', 0.0)
        if cpu_usage > self.resource_thresholds['cpu_usage_max']:
            violations.append(f"CPU使用率过高({cpu_usage:.1f}%)")
        
        # 检查内存使用率
        memory_usage = getattr(node, 'memory_usage', 0.0)
        if memory_usage > self.resource_thresholds['memory_usage_max']:
            violations.append(f"内存使用率过高({memory_usage:.1f}%)")
        
        # 检查电池电量（移动设备）
        battery_level = getattr(node, 'battery_level', 100.0)
        if battery_level < self.resource_thresholds['battery_level_min']:
            violations.append(f"电池电量过低({battery_level:.1f}%)")
        
        # 检查带宽
        bandwidth_factor = getattr(node, 'bandwidth_factor', 0.2)
        if bandwidth_factor < self.resource_thresholds['bandwidth_min'] / 100.0:
            violations.append(f"网络带宽不足({bandwidth_factor:.2f})")
        
        # 检查算力
        available_compute = self.estimate_compute_power(node)
        required_compute = compute_requirement * 1.2  # 保守估算，需要120%的算力
        if available_compute < required_compute:
            violations.append(f"算力不足(需要{required_compute:.1f},可用{available_compute:.1f} GFLOPS)")
        
        return len(violations) == 0, violations
    
    def select_nodes(self, 
                     available_nodes: List,
                     compute_requirement: Dict[str, Any],
                     num_primary_nodes: int = None) -> Dict[str, Any]:
        """
        选择主节点和备份节点
        
        Args:
            available_nodes: 可用节点列表
            compute_requirement: 算力需求字典
            num_primary_nodes: 需要的起始节点数（如果为None，则自动计算）
        
        Returns:
            选择结果，包含主节点和备份节点
        """
        required_compute = compute_requirement.get('total_compute', 0.0)
        required_memory = compute_requirement.get('memory_required', 0.0)
        required_gpu = compute_requirement.get('gpu_required', False)
        
        print(f"\n=== 节点选择开始 ===")
        print(f"算力需求: {required_compute:.2f} GFLOPS")
        print(f"内存需求: {required_memory:.2f} GB")
        print(f"需要GPU: {required_gpu}")
        print(f"可用节点数: {len(available_nodes)}")
        
        # 第一步：过滤满足基本约束的节点
        candidate_nodes = []
        for node in available_nodes:
            is_valid, violations = self.check_resource_constraints(node, required_compute)
            if is_valid:
                compute_power = self.estimate_compute_power(node)
                candidate_nodes.append((node, compute_power, violations))
            else:
                print(f"节点 {getattr(node, 'node_id', 'unknown')} 被过滤: {', '.join(violations)}")
        
        if len(candidate_nodes) == 0:
            return {
                'success': False,
                'message': '没有满足资源约束的节点',
                'primary_nodes': [],
                'backup_nodes': []
            }
        
        print(f"满足约束的候选节点数: {len(candidate_nodes)}")
        
        # 按算力排序
        candidate_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # 第二步：选择主节点
        # 估算需要的主节点数
        if num_primary_nodes is None:
            total_available_compute = sum(comp for _, comp, _ in candidate_nodes)
            num_primary_nodes = max(2, min(5, int(np.ceil(required_compute / 2000.0))))
            # 确保不超过可用节点数
            num_primary_nodes = min(num_primary_nodes, len(candidate_nodes))
        
        primary_nodes = []
        primary_compute = 0.0
        
        for node, compute_power, _ in candidate_nodes:
            if len(primary_nodes) >= num_primary_nodes:
                break
            primary_nodes.append(node)
            primary_compute += compute_power
            
            # 如果已经满足算力需求，可以提前停止
            if primary_compute >= required_compute * 1.1:
                break
        
        print(f"选择的主节点数: {len(primary_nodes)}")
        print(f"主节点总算力: {primary_compute:.2f} GFLOPS")
        
        # 第三步：选择备份节点
        remaining_nodes = [node for node, _, _ in candidate_nodes if node not in primary_nodes]
        backup_nodes = []
        
        # 至少选择min_backup_nodes个备份节点
        num_backup = max(self.min_backup_nodes, len(primary_nodes) // 2)
        num_backup = min(num_backup, len(remaining_nodes))
        
        for i, node in enumerate(remaining_nodes[:num_backup]):
            backup_nodes.append(node)
        
        print(f"选择的备份节点数: {len(backup_nodes)}")
        
        # 打印选中的节点信息
        print("\n主节点详情:")
        for i, node in enumerate(primary_nodes):
            compute = self.estimate_compute_power(node)
            node_id = getattr(node, 'node_id', 'unknown')
            gpu_name = getattr(node, 'gpu_name', 'N/A')
            print(f"  主节点{i+1}: {node_id}, GPU: {gpu_name}, 算力: {compute:.1f} GFLOPS")
        
        if backup_nodes:
            print("\n备份节点详情:")
            for i, node in enumerate(backup_nodes):
                compute = self.estimate_compute_power(node)
                node_id = getattr(node, 'node_id', 'unknown')
                gpu_name = getattr(node, 'gpu_name', 'N/A')
                print(f"  备份节点{i+1}: {node_id}, GPU: {gpu_name}, 算力: {compute:.1f} GFLOPS")
        
        print("=== 节点选择完成 ===\n")
        
        return {
            'success': True,
            'primary_nodes': primary_nodes,
            'backup_nodes': backup_nodes,
            'total_compute': primary_compute,
            'selection_details': {
                'num_candidates': len(candidate_nodes),
                'num_primary': len(primary_nodes),
                'num_backup': len(backup_nodes)
            }
        }
