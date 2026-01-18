"""
客户端示例
发送推理请求到边缘服务器
"""
import requests
from typing import Dict, Any, Optional


class AppClient:
    """App客户端 - 发送推理请求到边缘服务器"""
    
    def __init__(self, edge_server_url: str = "http://localhost:8080", **kwargs):
        """
        初始化App客户端
        
        Args:
            edge_server_url: 边缘服务器URL
            **kwargs: 其他参数（兼容InferenceClient）
        """
        # 兼容两种命名方式
        server_url = kwargs.get('server_url', edge_server_url)
        self.server_url = server_url.rstrip('/')
    
    def send_inference_request(self,
                              model_name: str,
                              input_data: Dict[str, Any],
                              model_source: str = "huggingface",
                              parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送推理请求（AppClient接口）
        
        Args:
            model_name: 模型名称
            input_data: 输入数据
            model_source: 模型来源
            parameters: 推理参数
        
        Returns:
            推理结果
        """
        return self.inference(model_name, input_data, model_source, parameters)
    
    def inference(self,
                  model_name: str,
                  input_data: Dict[str, Any],
                  model_source: str = "huggingface",
                  parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送推理请求
        
        Args:
            model_name: 模型名称
            input_data: 输入数据
            model_source: 模型来源（"huggingface"或"local"）
            parameters: 推理参数（batch_size等）
        
        Returns:
            推理结果
        """
        if parameters is None:
            parameters = {}
        
        url = f"{self.server_url}/api/inference"
        
        payload = {
            "model_name": model_name,
            "model_source": model_source,
            "input_data": input_data,
            "parameters": parameters
        }
        
        try:
            print(f"发送推理请求到: {url}")
            print(f"模型: {model_name} (来源: {model_source})")
            
            response = requests.post(url, json=payload, timeout=300)  # 5分钟超时
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                return {
                    'status': 'error',
                    'message': f"服务器错误: {response.status_code}",
                    'response': response.text
                }
        
        except requests.exceptions.Timeout:
            return {
                'status': 'error',
                'message': '请求超时'
            }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'error',
                'message': f'无法连接到服务器: {self.server_url}'
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            response = requests.get(f"{self.server_url}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.server_url}/api/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('models', [])
            return []
        except:
            return []


# 兼容别名
InferenceClient = AppClient


# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = AppClient(edge_server_url="http://localhost:8080")
    
    # 检查服务器健康状态
    if not client.health_check():
        print("服务器未响应，请确保边缘服务器已启动")
        exit(1)
    
    # 发送推理请求
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
    
    print("\n推理结果:")
    print(f"状态: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"使用的节点: {result.get('nodes_used', [])}")
        print(f"推理时间: {result.get('inference_time', 0):.2f} ms")
        print(f"结果: {result.get('result', {})}")
    else:
        print(f"错误: {result.get('message', 'Unknown error')}")
