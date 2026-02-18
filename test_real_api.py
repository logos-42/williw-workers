#!/usr/bin/env python3
"""测试实际的后端 API"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== 测试实际后端 API ===")
print()

# 测试导入
try:
    from edge_server.api_server import app
    print("✓ 成功导入 api_server")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 使用测试客户端
with app.test_client() as client:
    # 1. 健康检查
    print("1. GET /api/health")
    response = client.get('/api/health')
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {response.json}")
    print()
    
    # 2. 列出模型
    print("2. GET /api/models")
    response = client.get('/api/models')
    print(f"   状态码: {response.status_code}")
    print(f"   响应: {response.json}")
    print()
    
print("=== 测试完成 ===")