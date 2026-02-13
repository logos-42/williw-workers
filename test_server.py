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
