"""
完整流程演示
展示从app端发送请求到边缘服务器到调用算法到模型切分到分发出去的完整流程
"""
import sys
import os
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/work/lkc/williw-use')
sys.path.insert(0, '/work/lkc/youhua')

from edge_server.workflow_orchestrator import WorkflowOrchestrator
# 直接调用编排器，不需要通过API
# from interface_layer.app_client import AppClient
import numpy as np


def main():
    """主演示函数"""
    print("=" * 80)
    print("完整流程演示：App请求 -> 边缘服务器 -> 算法调用 -> 模型切分 -> 分发")
    print("=" * 80)
    print()
    
    # 步骤1: 模拟App客户端发送请求（直接调用编排器，不通过HTTP）
    print("【阶段1】App客户端发送推理请求")
    print("-" * 80)
    
    # 准备请求数据
    request_data = {
        'model_name': 'resnet18',  # 示例模型
        'model_source': 'huggingface',  # 或 'local'
        'input_data': {
            'input': np.random.randn(1, 3, 224, 224).astype(np.float32).tolist()
        },
        'parameters': {
            'batch_size': 1
        }
    }
    
    print(f"请求模型: {request_data['model_name']}")
    print(f"模型来源: {request_data['model_source']}")
    print(f"输入shape: {np.array(request_data['input_data']['input']).shape}")
    print()
    
    # 步骤2: 边缘服务器接收请求并处理（直接调用编排器）
    print("【阶段2】边缘服务器接收请求并开始处理")
    print("-" * 80)
    
    orchestrator = WorkflowOrchestrator()
    
    # 执行完整工作流
    try:
        result = orchestrator.execute_inference_workflow(
            model_name=request_data['model_name'],
            model_source=request_data['model_source'],
            input_data=request_data['input_data'],
            parameters=request_data['parameters']
        )
        
        # 步骤3: 显示结果
        print("\n【阶段3】处理结果")
        print("-" * 80)
        
        if result['status'] == 'success':
            print("✓ 推理任务执行成功！")
            print()
            
            print("详细信息:")
            print(f"  - 使用的节点数: {len(result.get('nodes_used', []))}")
            print(f"  - 节点ID: {result.get('nodes_used', [])}")
            print(f"  - 推理耗时: {result.get('inference_time', 0):.2f} ms")
            print(f"  - 算力需求: {result['compute_requirement']['total_compute']:.2f} GFLOPS")
            print(f"  - 内存需求: {result['compute_requirement']['memory_required']:.2f} GB")
            
            # 如果有算法详情，显示
            if 'algorithm_details' in result:
                alg_details = result['algorithm_details']
                print()
                print("算法执行详情:")
                print(f"  - 主节点数: {len(alg_details['node_selection']['primary_nodes'])}")
                print(f"  - 备份节点数: {len(alg_details['node_selection']['backup_nodes'])}")
                print(f"  - 路径优化路径数: {len(alg_details['routing']['routing_table'])}")
                print(f"  - 资源分配适应度: {alg_details['resource_allocation']['best_fitness']:.4f}")
                print(f"  - 分发模式: {alg_details['distribution_mode']}")
            
            print()
            print("模型信息:")
            model_info = result.get('model_info', {})
            print(f"  - 模型路径: {model_info.get('model_path', 'N/A')}")
            print(f"  - 模型格式: {model_info.get('format', 'N/A')}")
            if 'state_dict_path' in model_info:
                print(f"  - State dict路径: {model_info['state_dict_path']}")
            
            print()
            print("推理结果:")
            inference_result = result.get('result', {})
            if 'output_shape' in inference_result:
                print(f"  - 输出shape: {inference_result['output_shape']}")
            if 'output' in inference_result:
                output = inference_result['output']
                if isinstance(output, list):
                    print(f"  - 输出类型: list, 长度: {len(output)}")
                    if len(output) > 0:
                        print(f"  - 输出元素shape: {np.array(output[0]).shape if isinstance(output[0], list) else 'N/A'}")
            
            print()
            print("=" * 80)
            print("✓ 完整流程演示成功完成！")
            print("=" * 80)
            
        else:
            print("✗ 推理任务执行失败")
            print(f"错误信息: {result.get('message', '未知错误')}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n✗ 执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
