use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessRequest {
    pub model_id: String,
    pub metadata: ModelMetadata,
    pub nodes: Vec<NodeInfo>,
    /// optional: "compute" | "equal"
    pub strategy: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelMetadata {
    /// Ordered layers. layer_index must be 0..num_layers-1 and sorted ascending.
    pub layers: Vec<LayerMeta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayerMeta {
    pub layer_index: usize,
    pub name: Option<String>,
    pub compute_required: f64,
    pub num_params: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    /// Higher means stronger. Unit-free (relative ok).
    pub compute_power: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessResponse {
    pub model_id: String,
    pub num_layers: usize,
    pub plan: SplitPlan,
}

#[derive(Debug, Clone, Serialize)]
pub struct SplitPlan {
    /// Contiguous ranges by layer index (inclusive).
    pub assignments: Vec<NodeAssignment>,
    /// Execution order (chain). Node only needs this + its own range.
    pub execution_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeAssignment {
    pub node_id: String,
    pub start_layer: usize,
    pub end_layer: usize,
}

