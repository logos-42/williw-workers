"""
节点信息API
从williw-master获取节点信息，转换为Python对象
如果williw-master不可用，使用模拟数据
"""
import sys
import os
from typing import List, Dict, Any, Optional
import random

# 添加lkc项目路径
sys.path.insert(0, '/work/lkc/youhua')

try:
    from lkc.core.node import MobileNode, NetworkType, DeviceType, NodeStatus
except ImportError:
    # 如果lkc.core.node不可用，尝试从lkc.core导入
    try:
        from lkc.core import MobileNode, NetworkType, DeviceType, NodeStatus
    except ImportError:
        # 如果lkc项目不可用，定义基本结构
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
    """节点信息API（从williw-master获取或模拟）"""
    
    def __init__(self, williw_api_url: Optional[str] = None):
        """
        初始化节点信息API
        
        Args:
            williw_api_url: williw-master API地址（如果可用）
        """
        self.williw_api_url = williw_api_url
        self.use_mock = williw_api_url is None
    
    def get_available_nodes(self) -> List[MobileNode]:
        """
        获取可用节点列表
        
        Returns:
            节点列表
        """
        if self.use_mock or not self._check_williw_api():
            # 使用模拟数据
            return self._get_mock_nodes()
        else:
            # 从williw-master API获取
            return self._get_nodes_from_api()
    
    def _check_williw_api(self) -> bool:
        """检查williw-master API是否可用"""
        if not self.williw_api_url:
            return False
        try:
            import requests
            response = requests.get(f"{self.williw_api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _get_nodes_from_api(self) -> List[MobileNode]:
        """从williw-master API获取节点信息"""
        try:
            import requests
            response = requests.get(f"{self.williw_api_url}/api/nodes", timeout=5)
            if response.status_code == 200:
                nodes_data = response.json()
                nodes = []
                for node_data in nodes_data.get('nodes', []):
                    # 转换为MobileNode对象
                    node = self._convert_to_mobile_node(node_data)
                    nodes.append(node)
                return nodes
        except Exception as e:
            print(f"从williw-master获取节点信息失败: {str(e)}，使用模拟数据")
        
        return self._get_mock_nodes()
    
    def _convert_to_mobile_node(self, node_data: Dict[str, Any]) -> MobileNode:
        """将williw-master的节点数据转换为MobileNode对象"""
        # 根据williw-master的数据结构转换
        # 这里需要根据实际的API响应格式调整
        
        # 从device_capabilities获取信息
        capabilities = node_data.get('device_capabilities', {})
        
        # 从network获取信息
        network = node_data.get('network', {})
        
        # 从position获取地理位置
        position = node_data.get('position', {})
        
        node = MobileNode(
            node_id=node_data.get('node_id', 'unknown'),
            ip_address=node_data.get('address', '0.0.0.0'),
            location=node_data.get('location', '未知'),
            latitude=position.get('lat', 0.0),
            longitude=position.get('lon', 0.0),
            cpu_cores=capabilities.get('cpu_cores', 4),
            cpu_freq=capabilities.get('cpu_freq', 2.0),
            gpu_available=capabilities.get('has_gpu', False),
            gpu_memory=capabilities.get('gpu_memory_total_mb', 0) / 1024.0 if capabilities.get('gpu_memory_total_mb') else 0.0,
            cpu_usage=capabilities.get('cpu_usage', 0.0),
            gpu_usage=capabilities.get('gpu_usage_percent', 0.0),
            memory_usage=capabilities.get('memory_usage', 0.0),
            bandwidth_usage=network.get('bandwidth_usage', 0.0),
            battery_level=capabilities.get('battery_level', 100.0) * 100.0 if capabilities.get('battery_level') else 100.0,
            bandwidth=network.get('max_bandwidth_mbps', 10.0),
            network_latency=network.get('end_to_end_delay', 0.0) or 0.0,
            is_online=node_data.get('status') == 'Active',
            is_idle=node_data.get('available', True),
            reliability_score=node_data.get('reputation', 1.0),
            total_tasks_completed=node_data.get('completed_tasks', 0),
            total_tasks_failed=node_data.get('failed_tasks', 0)
        )
        
        return node
    
    def _get_mock_nodes(self, num_nodes: int = 20) -> List[MobileNode]:
        """生成模拟节点数据"""
        locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "西安", "武汉"]
        location_coords = {
            "北京": (39.9042, 116.4074),
            "上海": (31.2304, 121.4737),
            "广州": (23.1291, 113.2644),
            "深圳": (22.5431, 114.0579),
            "杭州": (30.2741, 120.1551),
            "成都": (30.6624, 104.0633),
            "西安": (34.3416, 108.9398),
            "武汉": (30.5928, 114.3055),
        }
        
        nodes = []
        for i in range(num_nodes):
            location = locations[i % len(locations)]
            coords = location_coords.get(location, (0.0, 0.0))
            
            # 随机生成节点能力
            cpu_cores = random.choice([4, 6, 8, 10, 12])
            cpu_freq = random.uniform(2.0, 3.5)
            gpu_available = random.choice([True, False, False])  # 33%概率有GPU
            gpu_memory = random.uniform(4.0, 8.0) if gpu_available else 0.0
            
            # 生成GPU信息
            gpu_names = ['RTX 4090', 'RTX 4080', 'RTX 3090', 'RTX 3080', 'RTX 3070', 'A100', 'T4']
            gpu_name = random.choice(gpu_names) if gpu_available else ''
            
            # 创建节点对象，添加算法需要的所有字段
            node_data = {
                'node_id': f"node_{i:03d}",
                'ip_address': f"192.168.1.{100 + i}",
                'location': location,
                'latitude': coords[0],
                'longitude': coords[1],
                'cpu_cores': cpu_cores,
                'cpu_freq': cpu_freq,
                'gpu_available': gpu_available,
                'gpu_memory': gpu_memory,
                'compute_power': (cpu_cores * cpu_freq * 2.0) + (gpu_memory * 50.0 if gpu_available else 0.0),
                'cpu_usage': random.uniform(5.0, 50.0),
                'gpu_usage': random.uniform(10.0, 60.0) if gpu_available else 0.0,
                'gpu_usage_percent': random.uniform(10.0, 60.0) if gpu_available else 0.0,
                'memory_usage': random.uniform(10.0, 60.0),
                'bandwidth_usage': random.uniform(5.0, 40.0),
                'battery_level': random.uniform(40.0, 95.0),
                'bandwidth': random.uniform(20.0, 100.0),
                'network_latency': random.uniform(10.0, 100.0),
                'is_online': True,
                'is_idle': True,
                'reliability_score': random.uniform(0.8, 1.0),
                'total_tasks_completed': random.randint(10, 100),
                'total_tasks_failed': random.randint(0, 5),
                # 添加算法需要的额外字段
                'gpu_name': gpu_name,
                'gpu_memory_used_mb': random.randint(1000, 8000) if gpu_available else 0,
                'gpu_memory_total_mb': int(gpu_memory * 1024) if gpu_available else 0,
                'max_memory_mb': random.randint(8192, 32768),
                'compute_capability': random.uniform(0.8, 1.2),
                'network_type': random.choice(['WiFi', 'Cellular5G', 'Cellular4G']),
                'bandwidth_factor': random.choice([1.0, 0.5, 0.3, 0.2]),
                'end_to_end_delay': random.uniform(10.0, 100.0),
                'distance_level': random.choice(['VeryClose', 'Close', 'Medium', 'Far']),
                'position_lat': coords[0],
                'position_lon': coords[1],
            }
            
            # 创建节点对象
            node = MobileNode(
                node_id=node_data['node_id'],
                ip_address=node_data['ip_address'],
                location=node_data['location'],
                latitude=node_data['latitude'],
                longitude=node_data['longitude'],
                cpu_cores=node_data['cpu_cores'],
                cpu_freq=node_data['cpu_freq'],
                gpu_available=node_data['gpu_available'],
                gpu_memory=node_data['gpu_memory'],
                compute_power=node_data['compute_power'],
                cpu_usage=node_data['cpu_usage'],
                gpu_usage=node_data['gpu_usage'],
                memory_usage=node_data['memory_usage'],
                bandwidth_usage=node_data['bandwidth_usage'],
                battery_level=node_data['battery_level'],
                bandwidth=node_data['bandwidth'],
                network_latency=node_data['network_latency'],
                is_online=node_data['is_online'],
                is_idle=node_data['is_idle'],
                reliability_score=node_data['reliability_score'],
                total_tasks_completed=node_data['total_tasks_completed'],
                total_tasks_failed=node_data['total_tasks_failed']
            )
            
            # 为节点对象动态添加额外属性
            for key, value in node_data.items():
                if not hasattr(node, key):
                    setattr(node, key, value)
            nodes.append(node)
        
        return nodes
