"""
边缘服务器API
接收app的推理请求，调用完整工作流
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from edge_server.workflow_orchestrator import WorkflowOrchestrator
import os

app = Flask(__name__)
CORS(app)

# 初始化工作流编排器
orchestrator = WorkflowOrchestrator()


@app.route('/api/inference', methods=['POST'])
def inference():
    """接收app的推理请求"""
    try:
        data = request.json
        
        # 验证请求数据
        if not data or 'model_name' not in data or 'input_data' not in data:
            return jsonify({
                'status': 'error',
                'message': '缺少必需参数: model_name, input_data'
            }), 400
        
        # 提取请求信息
        model_name = data['model_name']
        model_source = data.get('model_source', 'huggingface')
        input_data = data['input_data']
        parameters = data.get('parameters', {})
        
        print(f"\n{'='*70}")
        print(f"收到推理请求: {model_name} (来源: {model_source})")
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
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print(f"启动边缘服务器: http://{host}:{port}")
    app.run(host=host, port=port, debug=True)
