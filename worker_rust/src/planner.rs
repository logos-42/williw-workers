use anyhow::{anyhow, Result};

use crate::types::*;

pub fn build_plan(req: ProcessRequest) -> Result<ProcessResponse> {
    if req.nodes.is_empty() {
        return Err(anyhow!("nodes is empty"));
    }
    if req.metadata.layers.is_empty() {
        return Err(anyhow!("metadata.layers is empty"));
    }

    // validate layer_index monotonic and starting at 0 (best effort)
    let mut expected = 0usize;
    for l in &req.metadata.layers {
        if l.layer_index != expected {
            return Err(anyhow!(
                "layer_index must be contiguous starting at 0; expected {}, got {}",
                expected,
                l.layer_index
            ));
        }
        expected += 1;
    }

    let num_layers = req.metadata.layers.len();
    let strategy = req.strategy.clone().unwrap_or_else(|| "compute".to_string());

    let assignments = if strategy == "equal" {
        equal_contiguous_split(num_layers, &req.nodes)
    } else {
        compute_contiguous_split(&req.metadata, &req.nodes)
    }?;

    // Execution order: keep the same as assignments order (chain)
    let execution_order = assignments.iter().map(|a| a.node_id.clone()).collect();

    Ok(ProcessResponse {
        model_id: req.model_id,
        num_layers,
        plan: SplitPlan {
            assignments,
            execution_order,
        },
    })
}

fn equal_contiguous_split(num_layers: usize, nodes: &[NodeInfo]) -> Result<Vec<NodeAssignment>> {
    let n = nodes.len();
    let base = num_layers / n;
    let rem = num_layers % n;

    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    for (i, node) in nodes.iter().enumerate() {
        let mut count = base + if i < rem { 1 } else { 0 };
        if count == 0 {
            // If there are more nodes than layers, give empty ranges by collapsing to 0-length
            // but keep API stable by not emitting invalid ranges.
            continue;
        }
        let end = start + count - 1;
        out.push(NodeAssignment {
            node_id: node.node_id.clone(),
            start_layer: start,
            end_layer: end,
        });
        start = end + 1;
    }

    if out.is_empty() {
        return Err(anyhow!("no assignments produced"));
    }
    // If we skipped extra nodes, still must cover all layers
    if out.last().unwrap().end_layer != num_layers - 1 {
        // extend last to cover remainder
        let last = out.last_mut().unwrap();
        last.end_layer = num_layers - 1;
    }
    Ok(out)
}

fn compute_contiguous_split(meta: &ModelMetadata, nodes: &[NodeInfo]) -> Result<Vec<NodeAssignment>> {
    let num_layers = meta.layers.len();
    let total_compute: f64 = meta.layers.iter().map(|l| l.compute_required.max(0.0)).sum();
    if total_compute <= 0.0 {
        // fallback to equal split
        return equal_contiguous_split(num_layers, nodes);
    }

    let node_total: f64 = nodes.iter().map(|n| n.compute_power.max(0.0)).sum();
    if node_total <= 0.0 {
        return equal_contiguous_split(num_layers, nodes);
    }

    // Compute per-layer prefix sums in declared layer order (contiguous ranges).
    let mut prefix = Vec::with_capacity(num_layers + 1);
    prefix.push(0.0f64);
    for l in &meta.layers {
        let next = prefix.last().unwrap() + l.compute_required.max(0.0);
        prefix.push(next);
    }

    // Target cumulative compute for each cut.
    let mut targets = Vec::with_capacity(nodes.len());
    let mut acc = 0.0f64;
    for n in nodes {
        acc += n.compute_power.max(0.0) / node_total * total_compute;
        targets.push(acc);
    }
    // ensure last target is total_compute (avoid float drift)
    if let Some(last) = targets.last_mut() {
        *last = total_compute;
    }

    let mut out: Vec<NodeAssignment> = Vec::with_capacity(nodes.len());
    let mut start_idx = 0usize;

    for (node_i, node) in nodes.iter().enumerate() {
        if start_idx >= num_layers {
            break;
        }

        let target = targets[node_i];
        // find smallest end_idx such that prefix[end_idx+1] >= target
        let mut end_idx = start_idx;
        while end_idx + 1 < prefix.len() && prefix[end_idx + 1] < target {
            end_idx += 1;
        }

        // ensure progress: give at least one layer except possibly last when exhausted
        if end_idx < start_idx {
            end_idx = start_idx;
        }
        if node_i < nodes.len() - 1 && end_idx == num_layers - 1 && start_idx < num_layers - 1 {
            // avoid starving remaining nodes: keep at least one layer for later if possible
            end_idx = num_layers - 2;
        }

        out.push(NodeAssignment {
            node_id: node.node_id.clone(),
            start_layer: start_idx,
            end_layer: end_idx,
        });
        start_idx = end_idx + 1;
    }

    if out.is_empty() {
        return Err(anyhow!("no assignments produced"));
    }

    // cover all layers: extend last assignment
    if out.last().unwrap().end_layer != num_layers - 1 {
        let last = out.last_mut().unwrap();
        last.end_layer = num_layers - 1;
    }

    Ok(out)
}

