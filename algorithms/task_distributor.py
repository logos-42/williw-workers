"""
完整任务分发模块
实现megaphone mode（链式验证分发）
"""
from typing import Dict, List, Any, Optional
import time
import uuid


class TaskDistributor:
    """完整任务分发器 - Megaphone Mode"""
    
    def __init__(self):
        """初始化任务分发器"""
        self.task_status = {}  # {task_id: status}
        self.node_results = {}  # {task_id: {node_id: result}}
    
    def distribute_with_megaphone_mode(self,
                                      model_shards: List[Dict[str, Any]],
                                      nodes: List,
                                      input_data: Dict[str, Any],
                                      routing_table: Dict = None) -> Dict[str, Any]:
        """
        使用Megaphone Mode分发任务
        
        Megaphone Mode流程:
        1. Node A（起始节点）同时向B和C发送任务
        2. Node A执行自己的部分，然后传递剩余任务给B
        3. Node B执行任务并进行一致性验证，然后传递剩余任务给C
        4. 依此类推，形成链式传递
        
        Args:
            model_shards: 模型分片列表
            nodes: 节点列表
            input_data: 输入数据
            routing_table: 路由表
        
        Returns:
            分发结果
        """
        print(f"\n=== Megaphone Mode任务分发开始 ===")
        task_id = str(uuid.uuid4())
        print(f"任务ID: {task_id}")
        
        if len(model_shards) != len(nodes):
            raise ValueError(f"模型分片数({len(model_shards)})与节点数({len(nodes)})不匹配")
        
        # 初始化任务状态
        self.task_status[task_id] = {
            'status': 'distributing',
            'nodes': [getattr(n, 'node_id', f'node_{i}') if hasattr(n, 'node_id') else f'node_{i}' for i, n in enumerate(nodes)],
            'start_time': time.time()
        }
        self.node_results[task_id] = {}
        
        # Megaphone Mode分发
        # 第一步：起始节点（Node A）同时向B和C发送任务
        start_node_idx = 0
        if len(nodes) > 0:
            # 起始节点执行自己的分片
            start_node_id = getattr(nodes[start_node_idx], 'node_id', f'node_{start_node_idx}')
            print(f"\n步骤1: 起始节点 {start_node_idx} ({start_node_id}) 开始执行...")
            result_start = self._execute_shard(
                model_shards[start_node_idx],
                nodes[start_node_idx],
                input_data
            )
            self.node_results[task_id][start_node_id] = result_start
            print(f"  起始节点完成，输出shape: {result_start.get('output_shape', 'unknown')}")
            
            # 起始节点同时向后续节点发送任务（模拟并行发送）
            if len(nodes) > 1 and len(model_shards) > 1:
                print(f"\n步骤2: 起始节点同时向后续节点发送任务...")
                # 准备传递数据（包含前面节点的输出和验证信息）
                accumulated_output = result_start.get('output', None)
                verification_hash = self._compute_verification_hash(result_start)
                
                # 依次传递给后续节点
                for i in range(1, len(nodes)):
                    current_node = nodes[i]
                    current_shard = model_shards[i]
                    
                    node_id = getattr(current_node, 'node_id', f'node_{i}')
                    print(f"\n步骤{i+1}: 节点 {i} ({node_id}) 接收任务...")
                    
                    # 执行当前分片
                    result = self._execute_shard(
                        current_shard,
                        current_node,
                        accumulated_output if accumulated_output is not None else input_data
                    )
                    
                    # 一致性验证
                    if i > 1:  # 从第二个后续节点开始进行验证
                        print(f"  执行一致性验证...")
                        prev_node_id = getattr(nodes[i-1], 'node_id', f'node_{i-1}')
                        is_valid = self._verify_consistency(
                            result,
                            self.node_results[task_id].get(prev_node_id, {})
                        )
                        if not is_valid:
                            print(f"  警告: 一致性验证失败，但继续执行")
                    
                    self.node_results[task_id][node_id] = result
                    
                    # 更新累积输出（用于下一个节点）
                    accumulated_output = result.get('output', None)
                    verification_hash = self._compute_verification_hash(result)
                    
                    print(f"  节点 {i} 完成，输出shape: {result.get('output_shape', 'unknown')}")
        
        # 汇总结果
        total_time = time.time() - self.task_status[task_id]['start_time']
        
        print(f"\n所有节点执行完成，总耗时: {total_time:.2f}秒")
        print("=== Megaphone Mode任务分发完成 ===\n")
        
        self.task_status[task_id]['status'] = 'completed'
        self.task_status[task_id]['total_time'] = total_time
        
        return {
            'task_id': task_id,
            'status': 'completed',
            'node_results': self.node_results[task_id],
            'total_time': total_time,
            'distribution_mode': 'megaphone'
        }
    
    def _execute_shard(self,
                      shard: Dict[str, Any],
                      node: Any,
                      input_data: Any) -> Dict[str, Any]:
        """
        在节点上执行模型分片
        
        Args:
            shard: 模型分片信息
            node: 节点对象
            input_data: 输入数据
        
        Returns:
            执行结果
        """
        # 模拟执行（实际应该通过网络调用节点）
        time.sleep(0.1)  # 模拟执行时间
        
        # 模拟输出（实际应该是真实的模型推理结果）
        import numpy as np
        if isinstance(input_data, dict):
            # 如果是字典，假设有'input'键
            input_shape = input_data.get('input', np.array([1, 3, 224, 224])).shape
        else:
            input_shape = input_data.shape if hasattr(input_data, 'shape') else (1,)
        
        # 模拟输出（保持相同的batch size）
        output_shape = (input_shape[0], 1000)  # 假设输出是1000维
        
        return {
            'node_id': getattr(node, 'node_id', 'unknown'),
            'shard_path': shard.get('shard_path', ''),
            'output': np.random.randn(*output_shape).astype(np.float32),
            'output_shape': output_shape,
            'execution_time': 0.1,
            'layer_count': shard.get('layer_count', 0)
        }
    
    def _compute_verification_hash(self, result: Dict[str, Any]) -> str:
        """计算验证哈希"""
        import hashlib
        import pickle
        
        output = result.get('output', None)
        if output is not None:
            # 使用输出的统计信息计算哈希
            output_str = f"{output.shape}_{output.mean():.6f}_{output.std():.6f}"
            return hashlib.md5(output_str.encode()).hexdigest()
        return ''
    
    def _verify_consistency(self, current_result: Dict, previous_result: Dict) -> bool:
        """验证一致性"""
        # 简化的验证：检查输出维度是否合理
        current_shape = current_result.get('output_shape', ())
        previous_shape = previous_result.get('output_shape', ())
        
        if len(current_shape) == 0 or len(previous_shape) == 0:
            return False
        
        # 验证batch size是否一致
        if current_shape[0] != previous_shape[0]:
            return False
        
        return True
