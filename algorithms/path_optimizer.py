"""
路径优化模块
基于D-CACO算法的完整实现
"""
import sys
import os
sys.path.insert(0, '/work/lkc/youhua')

from typing import Dict, List, Tuple
import numpy as np
from lkc.algorithms.dcaco_algorithm import DCACO


class PathOptimizer:
    """路径优化器 - 使用D-CACO算法"""
    
    def __init__(self, num_nodes: int):
        """
        初始化路径优化器
        
        Args:
            num_nodes: 节点数量
        """
        self.num_nodes = num_nodes
        self.dcaco = DCACO(
            num_nodes=num_nodes,
            num_ants=100,
            max_iterations=200
        )
        self.routing_table = {}  # {(start, end): path}
    
    def build_graph_from_nodes(self, nodes: List) -> np.ndarray:
        """
        从节点列表构建网络图
        
        Args:
            nodes: 节点列表
        
        Returns:
            邻接矩阵
        """
        graph = np.zeros((self.num_nodes, self.num_nodes))
        
        # 获取节点位置信息
        node_positions = []
        for node in nodes:
            lat = getattr(node, 'position_lat', 0.0) if hasattr(node, 'position_lat') else 0.0
            lon = getattr(node, 'position_lon', 0.0) if hasattr(node, 'position_lon') else 0.0
            node_positions.append((lat, lon))
        
        # 构建连接矩阵（基于距离或网络延迟）
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    # 检查是否有直接连接（可以通过网络延迟判断）
                    # 这里简化处理，假设所有节点都能互相连接
                    graph[i, j] = 1.0
        
        return graph
    
    def extract_node_states(self, nodes: List) -> Dict[int, Dict]:
        """
        从节点列表提取节点状态
        
        Args:
            nodes: 节点列表
        
        Returns:
            节点状态字典
        """
        node_states = {}
        
        for idx, node in enumerate(nodes):
            state = {
                'U_cpu': getattr(node, 'cpu_usage', 0.0),
                'U_mem': getattr(node, 'memory_usage', 0.0),
                'U_net': getattr(node, 'bandwidth_usage', 0.0) if hasattr(node, 'bandwidth_usage') else 0.0,
                'D_e': getattr(node, 'end_to_end_delay', 50.0) if hasattr(node, 'end_to_end_delay') else 50.0,
                'P_ene': getattr(node, 'battery_level', 100.0) / 100.0
            }
            node_states[idx] = state
        
        return node_states
    
    def extract_link_states(self, nodes: List) -> Dict[Tuple[int, int], Dict]:
        """
        从节点列表提取链路状态
        
        Args:
            nodes: 节点列表
        
        Returns:
            链路状态字典
        """
        link_states = {}
        
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    # 计算链路延迟（基于地理位置和网络类型）
                    node_i = nodes[i]
                    node_j = nodes[j]
                    
                    # 获取网络类型和距离
                    network_type_i = getattr(node_i, 'network_type', 'Unknown')
                    distance_level_i = getattr(node_i, 'distance_level', 'Unknown')
                    
                    # 估算链路延迟（简化处理）
                    base_delay = 10.0  # 基础延迟（ms）
                    if distance_level_i == 'VeryClose':
                        base_delay = 5.0
                    elif distance_level_i == 'Close':
                        base_delay = 15.0
                    elif distance_level_i == 'Medium':
                        base_delay = 30.0
                    elif distance_level_i == 'Far':
                        base_delay = 100.0
                    
                    link_states[(i, j)] = {
                        'D_link': base_delay,
                        'P_loss': 0.01  # 1%丢包率（默认）
                    }
        
        return link_states
    
    def optimize_routing(self, nodes: List) -> Dict:
        """
        优化路由路径
        
        Args:
            nodes: 节点列表
        
        Returns:
            路由优化结果
        """
        print(f"\n=== 路径优化开始 ===")
        print(f"节点数量: {len(nodes)}")
        
        # 构建网络图
        graph = self.build_graph_from_nodes(nodes)
        
        # 提取节点状态
        node_states = self.extract_node_states(nodes)
        
        # 提取链路状态
        link_states = self.extract_link_states(nodes)
        
        # 更新D-CACO状态
        self.dcaco.update_node_states(node_states)
        self.dcaco.update_link_states(link_states)
        self.dcaco.update_graph(graph)
        
        # 为关键节点对优化路由
        routing_table = {}
        selected_pairs = []
        
        # 选择关键路径（从第一个节点到所有其他节点）
        if len(nodes) > 1:
            start_node = 0
            for end_node in range(1, len(nodes)):
                selected_pairs.append((start_node, end_node))
        
        print(f"优化路径数量: {len(selected_pairs)}")
        
        for start_idx, end_idx in selected_pairs:
            path, cost = self.dcaco.optimize_routing(
                start_idx, end_idx, node_states, link_states, graph
            )
            routing_table[(start_idx, end_idx)] = {
                'path': path,
                'cost': cost
            }
            print(f"路径 {start_idx} -> {end_idx}: {path}, 代价: {cost:.2f}")
        
        self.routing_table = routing_table
        
        print("=== 路径优化完成 ===\n")
        
        return {
            'routing_table': routing_table,
            'performance': self._calculate_performance(node_states, link_states)
        }
    
    def _calculate_performance(self, node_states: Dict, link_states: Dict) -> Dict:
        """计算路由性能指标"""
        if not self.routing_table:
            return {
                'avg_delay': 0.0,
                'avg_loss': 0.0
            }
        
        total_delay = 0.0
        total_loss = 0.0
        count = 0
        
        for (start, end), route_info in self.routing_table.items():
            path = route_info['path']
            if len(path) < 2:
                continue
            
            path_delay = 0.0
            path_loss = 1.0
            
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                if link in link_states:
                    path_delay += link_states[link]['D_link']
                    path_loss *= (1 - link_states[link]['P_loss'])
            
            total_delay += path_delay
            total_loss += (1 - path_loss)
            count += 1
        
        return {
            'avg_delay': total_delay / count if count > 0 else 0.0,
            'avg_loss': total_loss / count if count > 0 else 0.0
        }
