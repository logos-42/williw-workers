use worker::*;
use std::sync::Mutex;
use once_cell::sync::Lazy;

mod types;
mod planner;

// Global storage for received node info (in-memory)
static NODE_INFO_STORE: Lazy<Mutex<Vec<serde_json::Value>>> = Lazy::new(|| Mutex::new(Vec::new()));

#[event(fetch)]
pub async fn fetch(mut req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    // Basic router
    let url = req.url()?;
    let path = url.path();
    let method = req.method();

    // CORS headers helper
    let with_cors = |mut r: Response| -> Result<Response> {
        r.headers_mut()
            .set("Access-Control-Allow-Origin", "*")
            .ok();
        r.headers_mut()
            .set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            .ok();
        r.headers_mut()
            .set("Access-Control-Allow-Headers", "Content-Type")
            .ok();
        Ok(r)
    };

    // CORS preflight
    if method == Method::Options {
        return with_cors(Response::empty()?);
    }

    match (method, path) {
        // API Documentation - Root endpoint
        (Method::Get, "/") => {
            let doc = r#"<!DOCTYPE html>
<html>
<head>
    <title>Williw Workers API</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #000000; color: #ffffff; }
        h1 { color: #ffffff; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .endpoint { background: #0a0a0a; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #ffffff; }
        .method { display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
        .get { background: #ffffff; color: #000000; }
        .post { background: #333333; color: #ffffff; border: 1px solid #ffffff; }
        .path { font-family: monospace; font-size: 1.1em; color: #ffffff; }
        .desc { color: #888888; margin-top: 5px; }
        p { color: #aaaaaa; }
    </style>
</head>
<body>
    <h1>Williw Workers API</h1>
    <p>Decentralized training node coordination service</p>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/</span>
        <div class="desc">API Documentation</div>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/api/health</span>
        <div class="desc">Service health check</div>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/api/node-health?node_id=xxx</span>
        <div class="desc">Check node health status</div>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/api/nodes</span>
        <div class="desc">List all registered nodes</div>
    </div>
    
    <div class="endpoint">
        <span class="method get">GET</span>
        <span class="path">/api/messages?since=timestamp</span>
        <div class="desc">Poll messages from workers</div>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/node-info</span>
        <div class="desc">Upload node information</div>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/model</span>
        <div class="desc">Model selection request</div>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/request</span>
        <div class="desc">Inference request</div>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/training-data</span>
        <div class="desc">Upload training data</div>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/reassign-node</span>
        <div class="desc">Node reassignment</div>
    </div>
    
    <div class="endpoint">
        <span class="method post">POST</span>
        <span class="path">/api/process</span>
        <div class="desc">Compute strategy planning</div>
    </div>
</body>
</html>"#;
            let r = Response::from_html(doc)?;
            with_cors(r)
        }

        // Health check endpoint
        (Method::Get, "/api/health") => {
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Workers service is running",
                "service": "williw-workers"
            }))?;
            with_cors(r)
        }

        // List all registered nodes
        (Method::Get, "/api/nodes") => {
            let nodes = NODE_INFO_STORE.lock().unwrap();
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "count": nodes.len(),
                "nodes": nodes.clone()
            }))?;
            with_cors(r)
        }

        // Node info upload endpoint
        (Method::Post, "/api/node-info") => {
            let body: serde_json::Value = req.json().await.unwrap_or(serde_json::json!({}));
            console_log!("Received node info: {:?}", body);
            
            // Store the node info
            let mut nodes = NODE_INFO_STORE.lock().unwrap();
            nodes.push(body.clone());
            // Keep only last 100 nodes
            if nodes.len() > 100 {
                nodes.remove(0);
            }
            
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Node info received",
                "stored_count": nodes.len()
            }))?;
            with_cors(r)
        }

        // Model selection endpoint
        (Method::Post, "/api/model") => {
            let body: serde_json::Value = req.json().await.unwrap_or(serde_json::json!({}));
            console_log!("Received model selection: {:?}", body);
            
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Model selection received"
            }))?;
            with_cors(r)
        }

        // Inference request endpoint
        (Method::Post, "/api/request") => {
            let body: serde_json::Value = req.json().await.unwrap_or(serde_json::json!({}));
            console_log!("Received inference request: {:?}", body);
            
            // Return mock response for compatibility
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Inference request received",
                "request_id": format!("req_{}", js_sys::Math::random().to_string().replace(".", "")),
                "selected_nodes": [],
                "model_split_plan": {
                    "total_layers": 12,
                    "splits": [],
                    "communication_overhead": 0.0,
                    "estimated_inference_time": 1000
                },
                "estimated_total_time": 1000,
                "fallback_nodes": []
            }))?;
            with_cors(r)
        }

        // Training data upload endpoint
        (Method::Post, "/api/training-data") => {
            let body: serde_json::Value = req.json().await.unwrap_or(serde_json::json!({}));
            console_log!("Received training data: {:?}", body);
            
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Training data received"
            }))?;
            with_cors(r)
        }

        // Node reassignment endpoint
        (Method::Post, "/api/reassign-node") => {
            let body: serde_json::Value = req.json().await.unwrap_or(serde_json::json!({}));
            console_log!("Received reassignment request: {:?}", body);
            
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Node reassignment processed",
                "new_splits": [],
                "reassigned_nodes": []
            }))?;
            with_cors(r)
        }

        // Node health check endpoint
        (Method::Get, "/api/node-health") => {
            let node_id = url.query_pairs().find(|(k, _)| k == "node_id")
                .map(|(_, v)| v.to_string())
                .unwrap_or_default();
            
            let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "message": "Node health check",
                "node_id": node_id,
                "is_healthy": true,
                "last_seen": now,
                "current_load": 0.5,
                "issues": []
            }))?;
            with_cors(r)
        }

        // Messages polling endpoint
        (Method::Get, "/api/messages") => {
            let since = url.query_pairs().find(|(k, _)| k == "node_id")
                .map(|(_, v)| v.to_string());
            
            let now = js_sys::Date::new_0().to_iso_string().as_string().unwrap_or_default();
            // Return empty messages for now (no pending messages)
            let r = Response::from_json(&serde_json::json!({
                "success": true,
                "messages": [],
                "poll_timestamp": now
            }))?;
            with_cors(r)
        }

        // Legacy endpoints for compatibility
        (Method::Post, "/api/process") => {
            let body: types::ProcessRequest = req.json().await?;
            let plan = planner::build_plan(body).map_err(|e| worker::Error::RustError(e.to_string()))?;
            let r = Response::from_json(&plan)?;
            with_cors(r)
        }

        // Default: 404
        _ => {
            let r = Response::error(format!("Not found: {}", path), 404)?;
            with_cors(r)
        }
    }
}
