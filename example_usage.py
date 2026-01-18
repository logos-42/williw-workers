"""
Williw-Use 使用示例
演示如何启动边缘服务器和使用客户端
"""
import time
import multiprocessing
from edge_server.api_server import app
from interface_layer.app_client import InferenceClient


def run_server():
    """运行边缘服务器"""
    print("启动边缘服务器...")
    app.run(host='0.0.0.0', port=8080, debug=False)


def run_client_example():
    """运行客户端示例"""
    # 等待服务器启动
    time.sleep(3)
    
    # 创建客户端
    client = InferenceClient(server_url="http://localhost:8080")
    
    # 检查服务器健康状态
    if not client.health_check():
        print("❌ 服务器未响应")
        return
    
    print("✓ 服务器健康检查通过\n")
    
    # 示例1: 使用Hugging Face模型
    print("="*70)
    print("示例1: 使用Hugging Face模型进行推理")
    print("="*70)
    
    result = client.inference(
        model_name="bert-base-uncased",
        model_source="huggingface",
        input_data={
            "text": "Hello, world! This is a test."
        },
        parameters={
            "batch_size": 1
        }
    )
    
    print(f"\n结果状态: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"使用的节点: {result.get('nodes_used', [])}")
        print(f"推理时间: {result.get('inference_time', 0):.2f} ms")
        print(f"结果摘要: {result.get('result', {}).get('status', 'N/A')}")
    else:
        print(f"错误: {result.get('message', 'Unknown error')}")
    
    # 示例2: 使用本地模型
    print("\n" + "="*70)
    print("示例2: 使用本地模型进行推理")
    print("="*70)
    
    # 首先列出本地模型
    local_models = client.list_models()
    print(f"本地可用模型: {local_models}")
    
    if local_models:
        result = client.inference(
            model_name=local_models[0],
            model_source="local",
            input_data={
                "text": "Test input"
            }
        )
        print(f"\n结果状态: {result.get('status')}")


if __name__ == "__main__":
    print("Williw-Use 使用示例")
    print("="*70)
    print("注意: 这需要先安装依赖并配置模型仓库")
    print("="*70)
    print("\n选择运行模式:")
    print("1. 仅启动服务器")
    print("2. 仅运行客户端示例（需要服务器已运行）")
    print("3. 同时启动服务器和客户端")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        run_server()
    elif choice == "2":
        run_client_example()
    elif choice == "3":
        # 启动服务器进程
        server_process = multiprocessing.Process(target=run_server)
        server_process.start()
        
        try:
            # 运行客户端示例
            run_client_example()
        finally:
            # 停止服务器
            server_process.terminate()
            server_process.join()
    else:
        print("无效选择")
