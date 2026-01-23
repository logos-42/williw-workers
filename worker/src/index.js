/**
 * Cloudflare Worker - 完整流程处理
 * 1. 接收用户节点的模型请求
 * 2. 从 Hugging Face 读取元数据
 * 3. 执行算法（节点选择、资源分配、路径优化、按算力切分）
 * 4. 返回分配方案给用户节点
 */

// ====== 配置 ======
const METADATA_REPO = "logos42/williw";  // 元数据仓库
const HF_API_BASE = "https://huggingface.co";

// ====== 工具函数 ======

/**
 * 从 Hugging Face 读取元数据
 */
async function fetchMetadata(modelName, repo = null) {
  const metadataRepo = repo || METADATA_REPO;
  const filename = modelName.replace("/", "_") + "_metadata.json";
  const url = `${HF_API_BASE}/${metadataRepo}/resolve/main/${filename}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch metadata: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error fetching metadata: ${error.message}`);
    throw error;
  }
}

/**
 * 估算节点算力（GFLOPS）
 */
function estimateNodeCompute(node) {
  if (!node.gpu_available) {
    return (node.cpu_cores || 4) * 10.0;
  }
  
  const gpuMap = {
    'rtx 4090': 80000.0,
    'rtx 4080': 50000.0,
    'rtx 3090': 36000.0,
    'rtx 3080': 30000.0,
    'rtx 3070': 20000.0,
    'a100': 312000.0,
    'v100': 125000.0,
    't4': 8000.0,
    'k80': 5600.0,
  };
  
  const gpuName = (node.gpu_name || '').toLowerCase();
  let baseCompute = 5000.0;
  
  for (const [key, compute] of Object.entries(gpuMap)) {
    if (gpuName.includes(key)) {
      baseCompute = compute;
      break;
    }
  }
  
  // 考虑 GPU 使用率
  const gpuUsage = node.gpu_usage_percent || 0.0;
  return baseCompute * (1 - gpuUsage / 100.0);
}

/**
 * 节点选择算法（简化版）
 */
function selectNodes(availableNodes, computeRequirement) {
  // 按算力排序
  const nodesWithCompute = availableNodes.map(node => ({
    node,
    compute: estimateNodeCompute(node)
  })).sort((a, b) => b.compute - a.compute);
  
  // 选择满足算力需求的节点
  const selectedNodes = [];
  let totalCompute = 0;
  
  for (const { node, compute } of nodesWithCompute) {
    if (totalCompute < computeRequirement) {
      selectedNodes.push(node);
      totalCompute += compute;
    } else {
      break;
    }
  }
  
  return selectedNodes.length > 0 ? selectedNodes : [nodesWithCompute[0].node];
}

/**
 * 按算力切分模型（贪心算法）
 */
function computeBasedSplit(metadata, nodes) {
  const layers = metadata.layers;  // 每层信息（包含算力需求）
  const nodeComputePowers = nodes.map(n => estimateNodeCompute(n));
  
  // 按算力需求从高到低排序层
  const sortedLayers = [...layers].sort((a, b) => 
    b.compute_required - a.compute_required
  );
  
  // 初始化每个节点的算力使用情况
  const nodeUsage = nodes.map(n => ({
    node_id: n.node_id || n.id || `node_${nodes.indexOf(n)}`,
    compute_power: estimateNodeCompute(n),
    used_compute: 0.0,
    layer_names: []
  }));
  
  // 贪心分配：将算力需求高的层分配给算力高的节点
  for (const layer of sortedLayers) {
    // 找到可用算力最多的节点
    let bestNode = null;
    let bestRatio = -1;
    
    for (const node of nodeUsage) {
      const availableCompute = node.compute_power - node.used_compute;
      const ratio = availableCompute / layer.compute_required;
      
      if (ratio > bestRatio && ratio >= 1.0) {  // 确保节点能承担
        bestRatio = ratio;
        bestNode = node;
      }
    }
    
    // 如果所有节点都承担不了，分配给算力最高的节点
    if (!bestNode) {
      bestNode = nodeUsage.reduce((max, n) => 
        n.compute_power > max.compute_power ? n : max
      );
    }
    
    // 分配层给节点
    bestNode.layer_names.push(layer.name);
    bestNode.used_compute += layer.compute_required;
  }
  
  // 构建切分方案
  const splitPlan = {};
  for (const node of nodeUsage) {
    splitPlan[node.node_id] = {
      layer_names: node.layer_names,
      total_compute: node.used_compute,
      compute_utilization: node.used_compute / node.compute_power
    };
  }
  
  return splitPlan;
}

/**
 * 生成 Megaphone Mode 计划
 */
function generateMegaphonePlan(nodes, splitPlan) {
  if (nodes.length === 0) {
    return {};
  }
  
  const plan = {
    start_node: nodes[0].node_id || nodes[0].id || "node_0",
    chain: []
  };
  
  for (let i = 0; i < nodes.length; i++) {
    const nodeId = nodes[i].node_id || nodes[i].id || `node_${i}`;
    plan.chain.push({
      node_id: nodeId,
      step: i + 1,
      layer_names: splitPlan[nodeId]?.layer_names || [],
      next_nodes: i < nodes.length - 1 ? [nodes[i + 1].node_id || nodes[i + 1].id || `node_${i + 1}`] : []
    });
  }
  
  return plan;
}

// ====== 主处理函数 ======

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    
    // CORS 处理
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };
    
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }
    
    // ====== 根路径：健康信息 + API 名单 ======
    if (path === '/' || path === '') {
      return new Response(JSON.stringify({
        status: "healthy",
        message: "Williw Worker - 完整流程处理",
        timestamp: (new Date()).toISOString(),
        version: "2.0.0",
        apis: [
          {
            name: "request_inference",
            method: "POST",
            path: "/api/request",
            description: "用户节点发送模型请求（只接收请求，不处理）"
          },
          {
            name: "process_metadata",
            method: "POST",
            path: "/api/process",
            description: "用户节点通知 Worker 元数据已准备好，触发算法处理"
          },
          {
            name: "get_plan",
            method: "POST",
            path: "/api/get-plan",
            description: "用户节点获取分配方案"
          }
        ]
      }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" }
      });
    }
    
    // ====== API: 接收模型请求（只接收请求，不处理） ======
    if (path === '/api/request' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { model_name, node_id } = body;
        
        if (!model_name) {
          return new Response(JSON.stringify({
            error: "缺少 model_name 参数"
          }), {
            status: 400,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }
        
        console.log(`[Worker] 接收模型请求: ${model_name} from ${node_id}`);
        
        // 只返回确认，不处理
        const result = {
          success: true,
          message: "请求已接收，请上传元数据后调用 /api/process",
          model_name: model_name,
          node_id: node_id,
          received_at: (new Date()).toISOString()
        };
        
        return new Response(JSON.stringify(result), {
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
        
      } catch (error) {
        console.error(`[Worker] 错误: ${error.message}`);
        return new Response(JSON.stringify({
          error: error.message
        }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
    }
    
    // ====== API: 处理元数据并生成分配方案 ======
    if (path === '/api/process' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { model_name, node_id, metadata_repo } = body;
        
        if (!model_name) {
          return new Response(JSON.stringify({
            error: "缺少 model_name 参数"
          }), {
            status: 400,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }
        
        console.log(`[Worker] 开始处理模型: ${model_name} from ${node_id}`);
        
        // 1. 从 Hugging Face 读取元数据
        const repo = metadata_repo || env.METADATA_REPO || METADATA_REPO;
        console.log(`[Worker] 从仓库 ${repo} 读取元数据...`);
        const metadata = await fetchMetadata(model_name, repo);
        console.log(`[Worker] ✓ 元数据读取成功，总层数: ${metadata.total_layers}`);
        
        // 2. 获取可用节点（这里简化，实际应该从 D1/DO 获取）
        // TODO: 从 D1 或 DO 获取可用节点列表
        const availableNodes = body.available_nodes || [
          { node_id: "node_001", gpu_available: true, gpu_name: "RTX 3080", gpu_usage_percent: 20 },
          { node_id: "node_002", gpu_available: true, gpu_name: "RTX 3090", gpu_usage_percent: 30 },
          { node_id: "node_003", gpu_available: false, cpu_cores: 8 }
        ];
        
        // 3. 执行算法
        console.log(`[Worker] 执行算法...`);
        
        // 3.1 节点选择
        const computeRequirement = metadata.total_compute;
        const selectedNodes = selectNodes(availableNodes, computeRequirement);
        console.log(`[Worker] ✓ 节点选择完成，选中 ${selectedNodes.length} 个节点`);
        
        // 3.2 按算力切分
        const splitPlan = computeBasedSplit(metadata, selectedNodes);
        console.log(`[Worker] ✓ 按算力切分完成`);
        
        // 3.3 生成 Megaphone Mode 计划
        const megaphonePlan = generateMegaphonePlan(selectedNodes, splitPlan);
        console.log(`[Worker] ✓ Megaphone Mode 计划生成完成`);
        
        // 4. 保存方案到 D1/DO（可选）
        // TODO: 保存到 D1 或 DO，供后续查询
        const planId = `${model_name}_${node_id}_${Date.now()}`;
        
        // 5. 返回方案
        const result = {
          success: true,
          model_name: model_name,
          model_url: `https://huggingface.co/${model_name}`,
          metadata: {
            total_layers: metadata.total_layers,
            total_compute: metadata.total_compute,
            model_type: metadata.model_type
          },
          selected_nodes: selectedNodes.map(n => ({
            node_id: n.node_id || n.id,
            compute_power: estimateNodeCompute(n)
          })),
          split_plan: splitPlan,
          megaphone_plan: megaphonePlan,
          plan_id: planId,
          generated_at: (new Date()).toISOString()
        };
        
        console.log(`[Worker] ✓ 方案生成完成，plan_id: ${planId}`);
        
        return new Response(JSON.stringify(result), {
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
        
      } catch (error) {
        console.error(`[Worker] 错误: ${error.message}`);
        return new Response(JSON.stringify({
          error: error.message,
          stack: error.stack
        }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
    }
    
    // ====== API: 获取分配方案 ======
    if (path === '/api/get-plan' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { model_name, node_id } = body;
        
        if (!model_name || !node_id) {
          return new Response(JSON.stringify({
            error: "缺少 model_name 或 node_id 参数"
          }), {
            status: 400,
            headers: { ...corsHeaders, "Content-Type": "application/json" }
          });
        }
        
        // TODO: 从 D1 或 DO 读取已保存的方案
        // 这里简化，实际应该查询之前保存的方案
        
        return new Response(JSON.stringify({
          error: "方案未找到，请先调用 /api/request"
        }), {
          status: 404,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
        
      } catch (error) {
        return new Response(JSON.stringify({
          error: error.message
        }), {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
    }
    
    // ====== 404 ======
    return new Response(JSON.stringify({
      error: "Not Found"
    }), {
      status: 404,
      headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
};
