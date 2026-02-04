use serde::{Deserialize, Serialize};
use uuid::Uuid;
use worker::*;

#[derive(Debug, Deserialize)]
struct ApiRequestIn {
    model_id: String,
}

#[derive(Debug, Serialize)]
struct ApiRequestOut {
    ok: bool,
    request_id: Uuid,
}

#[derive(Debug, Deserialize)]
struct NodeInfoIn {
    node_id: String,
    /// 节点可用算力（抽象单位；建议用GFLOPS或任意一致单位）
    compute: f64,
}

#[derive(Debug, Deserialize)]
struct LayerMetaIn {
    layer_index: usize,
    /// 该层的计算量（抽象单位；建议是GFLOPs或某种 cost）
    compute_cost: f64,
}

#[derive(Debug, Deserialize)]
struct MetadataIn {
    layers: Vec<LayerMetaIn>,
}

#[derive(Debug, Deserialize)]
struct ApiProcessIn {
    model_id: String,
    nodes: Vec<NodeInfoIn>,
    metadata: MetadataIn,
}

#[derive(Debug, Serialize)]
struct LayerRangeOut {
    /// 起始层索引（包含）
    start_layer: usize,
    /// 结束层索引（包含）
    end_layer: usize,
}

#[derive(Debug, Serialize)]
struct NodeAssignmentOut {
    node_id: String,
    layer_range: LayerRangeOut,
}

/// **给节点的最小输出**：只包含顺序 + 每个节点负责的“第几层到第几层”
#[derive(Debug, Serialize)]
struct MinimalPlanOut {
    plan_version: u32,
    model_id: String,
    order: Vec<String>,
    assignments: Vec<NodeAssignmentOut>,
}

fn compute_based_layer_ranges(layers: &[LayerMetaIn], nodes: &[NodeInfoIn]) -> Vec<(usize, usize)> {
    if layers.is_empty() || nodes.is_empty() {
        return vec![];
    }

    // 按 layer_index 排序，确保稳定
    let mut sorted_layers: Vec<&LayerMetaIn> = layers.iter().collect();
    sorted_layers.sort_by_key(|l| l.layer_index);

    let total_cost: f64 = sorted_layers.iter().map(|l| l.compute_cost.max(0.0)).sum();
    let total_compute: f64 = nodes.iter().map(|n| n.compute.max(0.0)).sum();

    // 若无法按算力比例分配，则退化为均分（按层数）
    if total_cost <= 0.0 || total_compute <= 0.0 {
        let total_layers = sorted_layers.len();
        let base = total_layers / nodes.len();
        let rem = total_layers % nodes.len();
        let mut ranges = Vec::with_capacity(nodes.len());
        let mut cursor = 0usize;
        for i in 0..nodes.len() {
            let take = base + if i < rem { 1 } else { 0 };
            if take == 0 {
                ranges.push((0, 0));
                continue;
            }
            let start = cursor;
            let end = cursor + take - 1;
            cursor += take;
            ranges.push((sorted_layers[start].layer_index, sorted_layers[end].layer_index));
        }
        return ranges;
    }

    // 目标：每个节点拿到的 cost ≈ total_cost * (node_compute/total_compute)
    let targets: Vec<f64> = nodes
        .iter()
        .map(|n| total_cost * (n.compute.max(0.0) / total_compute))
        .collect();

    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(nodes.len());
    let mut cursor = 0usize;

    for (i, target) in targets.iter().enumerate() {
        if cursor >= sorted_layers.len() {
            // 没层了：给一个空范围（用最后一层索引占位）
            let last = sorted_layers.last().unwrap().layer_index;
            ranges.push((last, last));
            continue;
        }

        let start_idx = cursor;
        let mut acc = 0.0;

        // 确保每个节点至少拿到 1 层（除非已经没有层）
        while cursor < sorted_layers.len() {
            acc += sorted_layers[cursor].compute_cost.max(0.0);
            cursor += 1;

            // 最后一个节点直接拿剩余所有层，避免尾部漏分
            if i == nodes.len() - 1 {
                break;
            }

            if acc >= *target && (cursor - start_idx) >= 1 {
                break;
            }
        }

        let end_idx = cursor.saturating_sub(1).max(start_idx);
        ranges.push((
            sorted_layers[start_idx].layer_index,
            sorted_layers[end_idx].layer_index,
        ));
    }

    ranges
}

fn megaphone_order(nodes: &[NodeInfoIn]) -> Vec<String> {
    // 最小化：按输入顺序作为链式顺序（Node A -> Node B -> ...）
    nodes.iter().map(|n| n.node_id.clone()).collect()
}

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    console_error_panic_hook::set_once();

    let router = Router::new();

    router
        .post_async("/api/request", |_req, _ctx| async move {
            let input: ApiRequestIn = _req.json().await?;
            let _ = input; // 目前只做ack，不做存储
            Response::from_json(&ApiRequestOut {
                ok: true,
                request_id: Uuid::new_v4(),
            })
        })
        .post_async("/api/process", |_req, _ctx| async move {
            let input: ApiProcessIn = _req.json().await?;
            let ranges = compute_based_layer_ranges(&input.metadata.layers, &input.nodes);
            let order = megaphone_order(&input.nodes);

            let assignments: Vec<NodeAssignmentOut> = input
                .nodes
                .iter()
                .zip(ranges.into_iter())
                .map(|(n, (start, end))| NodeAssignmentOut {
                    node_id: n.node_id.clone(),
                    layer_range: LayerRangeOut {
                        start_layer: start,
                        end_layer: end,
                    },
                })
                .collect();

            let out = MinimalPlanOut {
                plan_version: 1,
                model_id: input.model_id,
                order,
                assignments,
            };

            Response::from_json(&out)
        })
        .run(req, env)
        .await
}

use worker::*;

mod types;
mod planner;

#[event(fetch)]
pub async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    // Basic router
    let url = req.url()?;
    let path = url.path();

    // CORS preflight
    if req.method() == Method::Options {
        return Response::empty()
            .map(|mut r| {
                r.headers_mut()
                    .set("Access-Control-Allow-Origin", "*")
                    .ok();
                r.headers_mut()
                    .set("Access-Control-Allow-Methods", "POST, OPTIONS")
                    .ok();
                r.headers_mut()
                    .set("Access-Control-Allow-Headers", "Content-Type")
                    .ok();
                r
            });
    }

    match (req.method(), path) {
        (Method::Post, "/api/request") => {
            // lightweight ack endpoint (kept for compatibility with older workflow)
            let mut r = Response::from_json(&serde_json::json!({
                "status": "ok",
                "message": "request accepted"
            }))?;
            r.headers_mut()
                .set("Access-Control-Allow-Origin", "*")
                .ok();
            Ok(r)
        }
        (Method::Post, "/api/process") => {
            let body: types::ProcessRequest = req.json().await?;
            let plan = planner::build_plan(body)?;
            Response::from_json(&plan).map(|mut r| {
                r.headers_mut()
                    .set("Access-Control-Allow-Origin", "*")
                    .ok();
                r
            })
        }
        _ => Response::error("not found", 404),
    }
}

