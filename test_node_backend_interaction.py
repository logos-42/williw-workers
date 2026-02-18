#!/usr/bin/env python3
"""
测试节点与后端 Workers 的交互
演示：
1. 启动模拟的边缘服务器
2. 节点发送数据到后端
3. 后端处理并返回结果
"""
import json
import time
import threading
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 模拟的节点数据（模拟 Rust 节点发送的数据）
MOCK_NODE_DATA = {
    "node_id": "node_001",
    "address": "192.168.1.100:9235",
    "device_capabilities": {
        "cpu_cores": 8,
        "cpu_freq": 3.0,
        "max_memory_mb": 16384,
        "has_gpu": True,
        "gpu_name": "Apple M1",
        "gpu_memory_total_mb": 8192,
        "gpu_usage_percent": 25.5,
        "battery_level": 0.85,
        "is_charging": True,
        "network_type": "WiFi",
        "bandwidth_factor": 1.0
    },
    "position": {
        "lat": 39.9042,
        "lon": 116.4074
    },
    "training_stats": {
        "tick_count": 1000,
        "model_version": 42,
        "model_hash": "0xabc123...",
        "convergence_score": 0.95,
        "sparse_updates_sent": 150,
        "sparse_updates_received": 200,
        "heartbeats_sent": 500,
        "heartbeats_received": 480
    },
    "topology_info": {
        "connected_peers": 3,
        "similarity": 0.85,
        "geo_affinity": 0.7,
        "network_affinity": 0.9
    },
    "consensus": {
        "stake_eth": 1.5,
        "stake_sol": 10.0,
        "reputation": 0.92
    },
    "status": "Active",
    "available": True,
    "last_active_at": int(time.time())
}

class MockEdgeServerHandler(BaseHTTPRequestHandler):
    """模拟边缘服务器处理器"""
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[Edge Server] {args[0]}")
    
    def _send_json_response(self, data, status=200):
        """发送 JSON 响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
    
    def do_GET(self):
        """处理 GET 请求"""
        if self.path == '/api/health':
            self._send_json_response({
                'status': 'healthy',
                'service': 'williw-edge-server-mock',
                'timestamp': time.time()
            })
        elif self.path == '/api/nodes':
            self._send_json_response({
                'status': 'success',
                'nodes': [MOCK_NODE_DATA],
                'total': 1
            })
        else:
            self._send_json_response({'error': 'Not found'}, 404)
    
    def do_POST(self):
        """处理 POST 请求"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json_response({'error': 'Invalid JSON'}, 400)
            return
        
        if self.path == '/api/node/register':
            # 节点注册
            print(f"\n📥 收到节点注册请求:")
            print(f"   节点ID: {data.get('node_id', 'unknown')}")
            print(f"   地址: {data.get('address', 'unknown')}")
            
            # 保存节点信息
            MOCK_NODE_DATA.update(data)
            
            self._send_json_response({
                'status': 'success',
                'message': 'Node registered successfully',
                'node_id': data.get('node_id'),
                'timestamp': time.time()
            })
            
        elif self.path == '/api/node/update':
            # 节点状态更新
            print(f"\n📊 收到节点状态更新:")
            print(f"   节点ID: {data.get('node_id', 'unknown')}")
            
            # 更新节点信息
            if 'device_capabilities' in data:
                print(f"   CPU使用率: {data['device_capabilities'].get('cpu_usage', 'N/A')}%")
                print(f"   GPU使用率: {data['device_capabilities'].get('gpu_usage_percent', 'N/A')}%")
                print(f"   电池: {data['device_capabilities'].get('battery_level', 'N/A')}")
            
            MOCK_NODE_DATA.update(data)
            
            self._send_json_response({
                'status': 'success',
                'message': 'Node status updated',
                'timestamp': time.time()
            })
            
        elif self.path == '/api/node/training':
            # 训练数据上报
            print(f"\n🔄 收到训练数据上报:")
            print(f"   节点ID: {data.get('node_id', 'unknown')}")
            
            if 'training_stats' in data:
                stats = data['training_stats']
                print(f"   训练轮次: {stats.get('tick_count', 'N/A')}")
                print(f"   收敛度: {stats.get('convergence_score', 'N/A')}")
                print(f"   模型版本: {stats.get('model_version', 'N/A')}")
            
            self._send_json_response({
                'status': 'success',
                'message': 'Training data received',
                'timestamp': time.time()
            })
            
        elif self.path == '/api/inference':
            # 推理请求
            print(f"\n🚀 收到推理请求:")
            print(f"   模型: {data.get('model_name', 'unknown')}")
            print(f"   来源: {data.get('model_source', 'unknown')}")
            
            self._send_json_response({
                'status': 'success',
                'message': 'Inference request accepted',
                'request_id': f"req_{int(time.time())}",
                'nodes_available': 1
            })
            
        else:
            self._send_json_response({'error': 'Endpoint not found'}, 404)


def run_mock_server(port=8080):
    """运行模拟服务器"""
    server = HTTPServer(('0.0.0.0', port), MockEdgeServerHandler)
    print(f"\n🌐 模拟边缘服务器启动: http://0.0.0.0:{port}")
    print("=" * 60)
    print("可用的 API 端点:")
    print(f"  GET  /api/health     - 健康检查")
    print(f"  GET  /api/nodes      - 获取节点列表")
    print(f"  POST /api/node/register  - 节点注册")
    print(f"  POST /api/node/update    - 节点状态更新")
    print(f"  POST /api/node/training  - 训练数据上报")
    print(f"  POST /api/inference      - 推理请求")
    print("=" * 60)
    server.serve_forever()


def test_node_to_backend_interaction():
    """测试节点与后端的交互"""
    server_url = "http://localhost:8080"
    
    print("\n" + "=" * 60)
    print("测试: 节点与后端 Workers 交互")
    print("=" * 60)
    
    # 等待服务器启动
    time.sleep(2)
    
    # 测试 1: 健康检查
    print("\n【测试 1】健康检查")
    try:
        response = requests.get(f"{server_url}/api/health", timeout=5)
        print(f"   状态: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        return
    
    # 测试 2: 节点注册
    print("\n【测试 2】节点注册")
    try:
        node_register_data = {
            "node_id": "rust_node_001",
            "address": "192.168.1.100:9235",
            "device_capabilities": {
                "cpu_cores": 8,
                "max_memory_mb": 16384,
                "has_gpu": True,
                "gpu_name": "Apple M1",
                "battery_level": 0.85,
                "network_type": "WiFi"
            },
            "position": {"lat": 39.9, "lon": 116.4}
        }
        response = requests.post(
            f"{server_url}/api/node/register",
            json=node_register_data,
            timeout=5
        )
        print(f"   状态: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 测试 3: 节点状态更新
    print("\n【测试 3】节点状态更新")
    try:
        update_data = {
            "node_id": "rust_node_001",
            "device_capabilities": {
                "cpu_usage": 35.5,
                "gpu_usage_percent": 42.0,
                "memory_usage": 45.2,
                "battery_level": 0.82,
                "is_charging": True
            }
        }
        response = requests.post(
            f"{server_url}/api/node/update",
            json=update_data,
            timeout=5
        )
        print(f"   状态: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 测试 4: 训练数据上报
    print("\n【测试 4】训练数据上报")
    try:
        training_data = {
            "node_id": "rust_node_001",
            "training_stats": {
                "tick_count": 1500,
                "model_version": 45,
                "convergence_score": 0.97,
                "sparse_updates_sent": 200,
                "sparse_updates_received": 280,
                "heartbeats_sent": 750,
                "heartbeats_received": 720
            },
            "topology_info": {
                "connected_peers": 4,
                "similarity": 0.88
            }
        }
        response = requests.post(
            f"{server_url}/api/node/training",
            json=training_data,
            timeout=5
        )
        print(f"   状态: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 测试 5: 获取节点列表
    print("\n【测试 5】获取节点列表")
    try:
        response = requests.get(f"{server_url}/api/nodes", timeout=5)
        print(f"   状态: {response.status_code}")
        result = response.json()
        print(f"   节点数: {result.get('total', 0)}")
        if result.get('nodes'):
            node = result['nodes'][0]
            print(f"   节点ID: {node.get('node_id')}")
            print(f"   状态: {node.get('status')}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    # 测试 6: 推理请求
    print("\n【测试 6】推理请求")
    try:
        inference_data = {
            "model_name": "bert-base-uncased",
            "model_source": "huggingface",
            "input_data": {"text": "Hello world"},
            "parameters": {"batch_size": 1}
        }
        response = requests.post(
            f"{server_url}/api/inference",
            json=inference_data,
            timeout=5
        )
        print(f"   状态: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试节点与后端交互')
    parser.add_argument('--server', action='store_true', help='仅启动服务器')
    parser.add_argument('--test', action='store_true', help='仅运行测试')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    
    args = parser.parse_args()
    
    if args.server:
        # 仅启动服务器
        run_mock_server(args.port)
    elif args.test:
        # 仅运行测试
        test_node_to_backend_interaction()
    else:
        # 同时启动服务器和测试
        server_thread = threading.Thread(
            target=run_mock_server,
            args=(args.port,),
            daemon=True
        )
        server_thread.start()
        
        # 运行测试
        test_node_to_backend_interaction()