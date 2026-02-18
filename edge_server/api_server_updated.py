"""
边缘服务器 API
接收 app 的推理请求，调用完整工作流
支持 iroh 节点注册
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from edge_server.workflow_orchestrator import WorkflowOrchestrator
from interface_layer.node_info_api import NodeInfoAPI
import os

app = Flask(__name__)
CORS(app)

# 初始化工作流编排器
orchestrator = WorkflowOrchestrator()

# 节点信息 API（用于管理 iroh 节点）
node_info_api = NodeInfoAPI()


@app.route('/api/inference', methods=['POST'])
def inference():
    """接收 app 的推理请求"""
    try:
        data = request.json

        # 验证请求数据
        if not data or 'model_name' not in data or 'input_data' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需参数：model_name, input_data'
            }), 400

        # 提取请求信息
        model_name = data['model_name']
        model_source = data.get('model_source', 'huggingface')
        input_data = data['input_data']
        parameters = data.get('parameters', {})

        print(f"\n{'='*70}")
        print(f"收到推理请求：{model_name} (来源：{model_source})")
        print(f"{'='*70}\n")

        # 执行完整工作流
        result = orchestrator.execute_inference_workflow(
            model_name=model_name,
            model_source=model_source,
            input_data=input_data,
            parameters=parameters
        )

        return jsonify(result)

    except Exception as e:
        print(f"错误：{str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/iroh-node/register', methods=['POST'])
def register_iroh_node():
    """
    注册 iroh 节点信息
    
    请求体:
    {
        "node_id": "iroh-node-id-xxx",
        "endpoint": "http://192.168.1.100:8080",
        "device_info": {
            "gpu_type": "cuda",
            "gpu_memory_total": 24,
            "cpu_cores": 8,
            ...
        },
        "iroh_node": {
            "node_id": "iroh-node-id-xxx",
            "addresses": ["..."],
            ...
        }
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': '请求体不能为空'
            }), 400
        
        # 注册节点
        success = node_info_api.register_iroh_node(data)
        
        if success:
            node_id = data.get('node_id') or data.get('device_id')
            return jsonify({
                'status': 'success',
                'message': f'iroh 节点注册成功',
                'node_id': node_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '节点注册失败'
            }), 500
            
    except Exception as e:
        print(f"iroh 节点注册错误：{str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/iroh-node/unregister/<node_id>', methods=['DELETE'])
def unregister_iroh_node(node_id: str):
    """注销 iroh 节点"""
    try:
        success = node_info_api.unregister_node(node_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'节点 {node_id} 已注销'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'节点 {node_id} 不存在'
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/nodes', methods=['GET'])
def get_nodes():
    """获取所有可用节点（包括 iroh 节点）"""
    try:
        nodes = node_info_api.get_available_nodes()
        
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                'node_id': node.node_id,
                'location': node.location,
                'gpu_available': node.gpu_available,
                'gpu_memory': node.gpu_memory,
                'compute_power': node.compute_power,
                'is_online': node.is_online,
                'is_idle': node.is_idle,
                'reliability_score': node.reliability_score
            })
        
        return jsonify({
            'status': 'success',
            'nodes': nodes_data,
            'total': len(nodes_data)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'service': 'williw-use-edge-server'
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """列出可用的模型（从本地仓库）"""
    try:
        models = orchestrator.model_fetcher.list_local_models()
        return jsonify({
            'status': 'success',
            'models': models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    print(f"启动边缘服务器：http://{host}:{port}")
    print(f"iroh 节点注册端点：http://{host}:{port}/api/iroh-node/register")
    app.run(host=host, port=port, debug=True)
