"""
集成检查脚本
检查接口层、算法层和边缘服务器是否完整集成
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, '/work/lkc/williw-use')
sys.path.insert(0, '/work/lkc/youhua')

print("=" * 80)
print("集成检查：接口层、算法层、边缘服务器")
print("=" * 80)
print()

errors = []
warnings = []

# 1. 检查接口层
print("【1】检查接口层...")
try:
    from interface_layer.app_client import AppClient
    print("  ✓ AppClient 导入成功")
    
    from interface_layer.node_info_api import NodeInfoAPI
    print("  ✓ NodeInfoAPI 导入成功")
    
    # 测试创建实例
    client = AppClient(edge_server_url="http://localhost:8080")
    print("  ✓ AppClient 实例创建成功")
    
    node_api = NodeInfoAPI()
    print("  ✓ NodeInfoAPI 实例创建成功")
    
    # 测试获取节点
    nodes = node_api.get_available_nodes()
    print(f"  ✓ 获取节点成功，共 {len(nodes)} 个节点")
    
except Exception as e:
    errors.append(f"接口层错误: {str(e)}")
    print(f"  ✗ 接口层错误: {str(e)}")

print()

# 2. 检查算法层
print("【2】检查算法层...")
try:
    from algorithms.task_scheduler import TaskScheduler
    print("  ✓ TaskScheduler 导入成功")
    
    from algorithms.node_selection import NodeSelector
    print("  ✓ NodeSelector 导入成功")
    
    from algorithms.resource_allocator import ResourceAllocator
    print("  ✓ ResourceAllocator 导入成功")
    
    from algorithms.path_optimizer import PathOptimizer
    print("  ✓ PathOptimizer 导入成功")
    
    from algorithms.model_splitter import ModelSplitter
    print("  ✓ ModelSplitter 导入成功")
    
    from algorithms.task_distributor import TaskDistributor
    print("  ✓ TaskDistributor 导入成功")
    
    # 测试创建实例
    scheduler = TaskScheduler()
    print("  ✓ TaskScheduler 实例创建成功")
    
    selector = NodeSelector()
    print("  ✓ NodeSelector 实例创建成功")
    
except Exception as e:
    errors.append(f"算法层错误: {str(e)}")
    print(f"  ✗ 算法层错误: {str(e)}")
    import traceback
    traceback.print_exc()

print()

# 3. 检查边缘服务器
print("【3】检查边缘服务器...")
try:
    from edge_server.workflow_orchestrator import WorkflowOrchestrator
    print("  ✓ WorkflowOrchestrator 导入成功")
    
    from edge_server.api_server import app
    print("  ✓ Flask API Server 导入成功")
    
    from edge_server.model_fetcher import ModelFetcher
    print("  ✓ ModelFetcher 导入成功")
    
    from edge_server.model_converter import ModelConverter
    print("  ✓ ModelConverter 导入成功")
    
    from edge_server.compute_estimator import ComputeEstimator
    print("  ✓ ComputeEstimator 导入成功")
    
    # 测试创建实例
    orchestrator = WorkflowOrchestrator()
    print("  ✓ WorkflowOrchestrator 实例创建成功")
    
    if orchestrator.task_scheduler is not None:
        print("  ✓ 完整算法层已加载")
    else:
        warnings.append("完整算法层未加载，使用简化版本")
        print("  ⚠ 完整算法层未加载，使用简化版本")
    
except Exception as e:
    errors.append(f"边缘服务器错误: {str(e)}")
    print(f"  ✗ 边缘服务器错误: {str(e)}")
    import traceback
    traceback.print_exc()

print()

# 4. 检查lkc基础模块
print("【4】检查lkc基础模块...")
try:
    from lkc.core.node import MobileNode
    print("  ✓ MobileNode 导入成功")
    
    from lkc.algorithms.dcaco_algorithm import DCACO
    print("  ✓ DCACO 导入成功")
    
    # 测试NL-APSO
    try:
        from nl_apso import NLAPSO
        print("  ✓ NL-APSO 导入成功")
    except ImportError:
        warnings.append("NL-APSO导入失败，可能需要检查路径")
        print("  ⚠ NL-APSO 导入失败（可能需要检查路径）")
    
except Exception as e:
    errors.append(f"lkc基础模块错误: {str(e)}")
    print(f"  ✗ lkc基础模块错误: {str(e)}")

print()

# 5. 检查完整流程集成
print("【5】检查完整流程集成...")
try:
    from edge_server.workflow_orchestrator import WorkflowOrchestrator
    from interface_layer.node_info_api import NodeInfoAPI
    
    orchestrator = WorkflowOrchestrator()
    node_api = NodeInfoAPI()
    
    # 检查工作流编排器是否能访问所有组件
    assert orchestrator.model_fetcher is not None, "ModelFetcher未初始化"
    assert orchestrator.model_converter is not None, "ModelConverter未初始化"
    assert orchestrator.compute_estimator is not None, "ComputeEstimator未初始化"
    assert orchestrator.node_info_api is not None, "NodeInfoAPI未初始化"
    
    print("  ✓ 工作流编排器组件完整")
    
    # 检查算法层是否可用
    if orchestrator.task_scheduler is not None:
        scheduler = orchestrator.task_scheduler
        assert scheduler.node_selector is not None, "NodeSelector未初始化"
        assert scheduler.model_splitter is not None, "ModelSplitter未初始化"
        assert scheduler.task_distributor is not None, "TaskDistributor未初始化"
        print("  ✓ 完整算法层组件完整")
    else:
        warnings.append("完整算法层未加载")
        print("  ⚠ 完整算法层未加载")
    
except Exception as e:
    errors.append(f"流程集成错误: {str(e)}")
    print(f"  ✗ 流程集成错误: {str(e)}")
    import traceback
    traceback.print_exc()

print()
print("=" * 80)
print("集成检查结果")
print("=" * 80)

if len(errors) == 0:
    print("✓ 所有核心模块已成功集成！")
    if warnings:
        print(f"\n⚠ 警告 ({len(warnings)} 个):")
        for w in warnings:
            print(f"  - {w}")
    print("\n✓ 接口层、算法层、边缘服务器已完整集成")
    print("✓ 可以运行完整流程演示（full_demo.py）")
    exit_code = 0
else:
    print(f"✗ 发现 {len(errors)} 个错误:")
    for e in errors:
        print(f"  - {e}")
    exit_code = 1

if warnings:
    print(f"\n⚠ 警告 ({len(warnings)} 个):")
    for w in warnings:
        print(f"  - {w}")

print("=" * 80)
sys.exit(exit_code)
