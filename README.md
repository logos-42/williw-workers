# Williw-Use 分布式模型推理系统

整合三个项目的完整分布式模型推理系统：
- **lkc** - 算法层（Python）：节点选择、路径优化、资源分配、模型切分
- **williw-master** - 接口层（Rust/Tauri + TypeScript）：节点信息提供
- **边缘服务器** - 模型获取、转换、推理管理

## 完整流程

```
接口层（app/客户端）
    ↓ HTTP POST请求
    {
        "model_name": "bert-base-uncased",
        "model_source": "huggingface",
        "input_data": {...},
        "parameters": {"batch_size": 1}
    }
    ↓
边缘服务器（Edge Server API）
    ↓ 1. 从Hugging Face/模型仓库获取模型
    ↓ 2. ONNX → PyTorch转换（如果需要）
    ↓ 3. 读取state_dict文件
    ↓ 4. 估算模型算力需求（保守估算，可算多不可算少）
    ↓ 5. 从接口层（williw-master）读取节点信息
    ↓ 6. 调用算法层（lkc）:
        - 节点选择（NodeSelectionAlgorithm）
        - 路径优化（D-CACO蚁群算法）
        - 资源分配（LCR-AGA遗传算法 + NL-APSO粒子群）
        - 模型切分（ModelSplitter）
    ↓ 7. 分布式模型推理（非训练）
    ↓ 8. 集成推理结果
    ↓ HTTP响应
    {
        "status": "success",
        "result": {...},
        "nodes_used": [...],
        "inference_time": 123.4
    }
    ↓
接口层（app/客户端）显示结果
```

## 项目结构

```
williw-use/
├── edge_server/              # 边缘服务器
│   ├── __init__.py
│   ├── api_server.py         # Flask API服务器
│   ├── model_fetcher.py      # 模型获取（Hugging Face/本地）
│   ├── model_converter.py    # ONNX → PyTorch转换
│   ├── compute_estimator.py  # 模型算力估算（保守）
│   ├── inference_manager.py  # 推理管理器
│   └── workflow_orchestrator.py  # 工作流编排器
├── interface_layer/          # 接口层（从williw-master读取节点信息）
│   ├── __init__.py
│   ├── node_info_client.py   # 节点信息客户端（调用williw-master API）
│   └── app_client.py         # 客户端示例（发送推理请求）
├── models/                   # 模型相关（复用lkc，扩展推理功能）
│   ├── __init__.py
│   ├── inference_engine.py   # 分布式推理引擎
│   └── result_merger.py      # 结果集成
├── algorithms/               # 算法层（复用lkc）
│   └── (链接到lkc/algorithms)
├── utils/                    # 工具函数
│   ├── __init__.py
│   └── config.py             # 配置管理
├── requirements.txt
└── README.md
```

## 安装和使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml` 或设置环境变量：
- `WILLIW_API_URL`: williw-master接口层API地址
- `HUGGINGFACE_CACHE_DIR`: Hugging Face模型缓存目录
- `MODEL_REPOSITORY_PATH`: 本地模型仓库路径

### 3. 启动边缘服务器

```bash
python -m edge_server.api_server --port 8080
```

### 4. 客户端使用示例

```python
from interface_layer.app_client import InferenceClient

client = InferenceClient(server_url="http://localhost:8080")
result = client.inference(
    model_name="bert-base-uncased",
    model_source="huggingface",
    input_data={"text": "Hello world"},
    parameters={"batch_size": 1}
)
print(result)
```

## 核心功能

### 1. 模型获取
- 从Hugging Face Hub下载模型
- 从本地模型仓库加载模型
- 支持ONNX和PyTorch格式

### 2. 模型转换
- ONNX → PyTorch转换
- 读取state_dict文件

### 3. 算力估算（保守策略）
- 基于模型结构和参数数量估算算力需求
- 安全系数1.5（可算多不可算少）
- 考虑激活值开销和内存访问开销

### 4. 节点信息获取
- 从williw-master接口层读取节点信息
- 转换为lkc的MobileNode对象

### 5. 算法调用（lkc）
- 节点选择：基于距离、能力、可靠性
- 路径优化：D-CACO蚁群算法
- 资源分配：LCR-AGA + NL-APSO
- 模型切分：按层切分

### 6. 分布式推理
- 链式推理：每个节点处理自己的层
- 激活值在节点间传递
- 最终结果从最后一个节点返回

### 7. 结果集成
- 合并各节点的推理结果
- 处理模型切分边界的结果拼接

## 依赖

- Flask（边缘服务器API）
- torch, transformers（模型处理）
- onnx, onnxruntime（模型转换）
- requests（API调用）
- lkc项目（算法层）
