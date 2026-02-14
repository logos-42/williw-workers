"""
简化的测试 API 服务器
用于测试 Tauri 客户端连接
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "success": True,
        "message": "Test server is running",
        "service": "williw-workers-test"
    })

@app.route('/api/node-info', methods=['POST'])
def node_info():
    data = request.json or {}
    print(f"Received node info: {data}")
    return jsonify({
        "success": True,
        "message": "Node info received"
    })

@app.route('/api/model', methods=['POST'])
def model():
    data = request.json or {}
    print(f"Received model: {data}")
    return jsonify({
        "success": True,
        "message": "Model selection received"
    })

@app.route('/api/request', methods=['POST'])
def request_inference():
    data = request.json or {}
    print(f"Received inference request: {data}")
    return jsonify({
        "success": True,
        "message": "Inference request received",
        "request_id": f"req_{datetime.now().timestamp()}",
        "selected_nodes": [],
        "model_split_plan": {
            "total_layers": 12,
            "splits": [],
            "communication_overhead": 0.0,
            "estimated_inference_time": 1000
        },
        "estimated_total_time": 1000,
        "fallback_nodes": []
    })

@app.route('/api/process', methods=['POST'])
def process_compute():
    """模型切分规划端点 - 真实算法实现"""
    data = request.json or {}
    print(f"Received process request: {data}")
    
    # 提取参数
    model_info = data.get("model", {})
    nodes = data.get("nodes", [])
    strategy = data.get("strategy", "compute")
    
    # 获取模型层数（如果没有提供，使用默认值）
    total_layers = model_info.get("total_layers", 16)
    
    # 计算切分方案
    num_nodes = len(nodes) if nodes else 1
    layers_per_node = max(1, total_layers // max(num_nodes, 1))
    
    splits = []
    for i, node in enumerate(nodes):
        start_layer = i * layers_per_node
        end_layer = min(start_layer + layers_per_node, total_layers)
        
        splits.append({
            "node_id": node.get("node_id", f"node_{i}"),
            "layer_range": [start_layer, end_layer],
            "layers": list(range(start_layer, end_layer)),
            "estimated_compute_time": (end_layer - start_layer) * 100,
        })
    
    # 如果没有节点，返回空方案
    if not nodes:
        splits = [{"node_id": "local", "layer_range": [0, total_layers], "layers": list(range(total_layers))}]
    
    # 计算通信开销（基于节点数量）
    communication_overhead = len(nodes) * 10 if len(nodes) > 1 else 0
    
    # 估算总时间
    estimated_time = sum(s.get("estimated_compute_time", 0) for s in splits) + communication_overhead
    
    return jsonify({
        "success": True,
        "strategy": strategy,
        "model": model_info,
        "total_nodes": num_nodes,
        "total_layers": total_layers,
        "splits": splits,
        "communication_overhead": communication_overhead,
        "estimated_total_time": estimated_time,
        "message": f"Generated split plan for {num_nodes} nodes"
    })

@app.route('/api/training-data', methods=['POST'])
def training_data():
    data = request.json or {}
    print(f"Received training data: {data}")
    return jsonify({
        "success": True,
        "message": "Training data received"
    })

@app.route('/api/reassign-node', methods=['POST'])
def reassign_node():
    data = request.json or {}
    print(f"Received reassignment: {data}")
    return jsonify({
        "success": True,
        "message": "Node reassignment processed",
        "new_splits": [],
        "reassigned_nodes": []
    })

@app.route('/api/node-health', methods=['GET'])
def node_health():
    node_id = request.args.get('node_id', '')
    return jsonify({
        "success": True,
        "message": "Node health check",
        "node_id": node_id,
        "is_healthy": True,
        "last_seen": datetime.now().isoformat(),
        "current_load": 0.5,
        "issues": []
    })

@app.route('/api/messages', methods=['GET'])
def messages():
    since = request.args.get('since')
    return jsonify({
        "success": True,
        "messages": [],
        "poll_timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting test server on http://0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
