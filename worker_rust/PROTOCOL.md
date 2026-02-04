## 最小协议（给节点的输出只包含：顺序 + 第几层到第几层）

### 目标

- Worker/DO 侧只做**算法规划**（不下载/不解析 state_dict）
- 节点侧负责：下载模型、提取元数据、按范围切分与分发
- Worker 输出对节点尽量“少信息”：只需要知道自己负责的**层索引范围**与**执行顺序**

### `/api/process` 请求

字段说明：
- `model_id`: 模型标识（字符串）
- `metadata.layers`: 按顺序排列的层信息；`layer_index` 必须从 0 连续递增
- `nodes`: 节点算力信息（相对值即可）
- `strategy`: `"compute"`（默认）或 `"equal"`

请求示例见：`node_client/demo_process_request.json`

### `/api/process` 响应（精简输出）

字段说明：
- `plan.assignments`: 每个节点一个**连续范围**（`start_layer/end_layer` inclusive）
- `plan.execution_order`: 链式执行顺序（通常与 assignments 顺序一致）

响应示例见：`node_client/demo_process_response.json`

