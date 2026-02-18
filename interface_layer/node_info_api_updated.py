"""
节点信息 API
从 williw-master 获取节点信息，转换为 Python 对象
如果 williw-master 不可用，使用模拟数据
支持 iroh 节点注册
"""
import sys
import os
from typing import List, Dict, Any, Optional
import random
from datetime import datetime

# 添加 lkc 项目路径
sys.path.insert(0, '/work/lkc/youhua')

try:
    from lkc.core.node import MobileNode, NetworkType, DeviceType, NodeStatus
except ImportError:
    try:
        from lkc.core import MobileNode, NetworkType, DeviceType, NodeStatus
    except ImportError:
        from dataclasses import dataclass
        from enum import Enum

        class NetworkType(Enum):
            WIFI = "WiFi"
            CELLULAR_4G = "Cellular4G"
            CELLULAR_5G = "Cellular5G"
            UNKNOWN = "Unknown"

        class DeviceType(Enum):
            PHONE = "Phone"
            TABLET = "Tablet"
            DESKTOP = "Desktop"
            UNKNOWN = "Unknown"

        class NodeStatus(Enum):
            ACTIVE = "Active"
            OFFLINE = "Offline"
            BUSY = "Busy"
            PAUSED = "Paused"

        @dataclass
        class MobileNode:
            node_id: str
            ip_address: str
            location: str = "未知"
            latitude: float = 0.0
            longitude: float = 0.0
            cpu_cores: int = 4
            cpu_freq: float = 2.0
            gpu_available: bool = False
            gpu_memory: float = 0.0
            compute_power: float = 0.0
            cpu_usage: float = 0.0
            gpu_usage: float = 0.0
            memory_usage: float = 0.0
            bandwidth_usage: float = 0.0
            battery_level: float = 100.0
            bandwidth: float = 10.0
            network_latency: float = 0.0
            is_online: bool = True
            is_idle: bool = True
            current_task_id: Optional[str] = None
            reliability_score: float = 1.0
            total_tasks_completed: int = 0
            total_tasks_failed: int = 0


class NodeInfoAPI:
    """节点信息 API（从 williw-master 获取或模拟）"""

    def __init__(self, williw_api_url: Optional[str] = None):
        """
        初始化节点信息 API

        Args:
            williw_api_url: williw-master API 地址（如果可用）
        """
        self.williw_api_url = williw_api_url
        self.use_mock = williw_api_url is None
        # 存储已注册的节点（包括 iroh 节点）
        self.registered_nodes: Dict[str, Dict[str, Any]] = {}

    def get_available_nodes(self) -> List[MobileNode]:
        """
        获取可用节点列表

        Returns:
            节点列表
        """
        # 1. 优先使用已注册的节点（iroh 节点）
        if self.registered_nodes:
            return self._get_registered_nodes()
        
        # 2. 从 williw-master API 获取
        if not self.use_mock and self._check_williw_api():
            return self._get_nodes_from_api()
        
        # 3. 使用模拟数据
        return self._get_mock_nodes()

    def register_iroh_node(self, node_data: Dict[str, Any]) -> bool:
        """
        注册 iroh 节点信息

        Args:
            node_data: 节点数据，包含：
                - node_id: iroh 节点 ID
                - endpoint: 节点地址
                - device_info: 设备信息
                - iroh_node: iroh 节点详细信息

        Returns:
            是否注册成功
        """
        try:
            node_id = node_data.get('node_id') or node_data.get('device_id')
            if not node_id:
                print(f"注册失败：缺少 node_id")
                return False

            # 转换为 MobileNode
            mobile_node = self._convert_to_mobile_node(node_data)
            
            # 存储节点信息
            self.registered_nodes[node_id] = {
                'node': mobile_node,
                'raw_data': node_data,
                'registered_at': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'is_iroh_node': True
            }

            print(f"✅ iroh 节点注册成功：{node_id}")
            return True

        except Exception as e:
            print(f"❌ iroh 节点注册失败：{str(e)}")
            return False

    def _get_registered_nodes(self) -> List[MobileNode]:
        """获取已注册的节点列表"""
        nodes = []
        current_time = datetime.now()

        for node_id, node_info in self.registered_nodes.items():
            # 检查节点是否过期（30 分钟未更新）
            last_seen = datetime.fromisoformat(node_info['last_seen'])
            if (current_time - last_seen).total_seconds() > 1800:
                print(f"⚠️ 节点 {node_id} 已过期，移除")
                continue

            node = node_info['node']
            # 更新最后活跃时间
            node_info['last_seen'] = current_time.isoformat()
            nodes.append(node)

        print(f"📊 获取到 {len(nodes)} 个已注册节点")
        return nodes

    def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        if node_id in self.registered_nodes:
            del self.registered_nodes[node_id]
            print(f"✅ 节点 {node_id} 已注销")
            return True
        return False

    def get_node_by_id(self, node_id: str) -> Optional[MobileNode]:
        """根据节点 ID 获取节点"""
        if node_id in self.registered_nodes:
            return self.registered_nodes[node_id]['node']
        return None
