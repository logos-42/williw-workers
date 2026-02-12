use worker::*;

mod types;
mod planner;

#[event(fetch)]
pub async fn fetch(mut req: Request, _env: Env, _ctx: Context) -> Result<Response> {
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
            let plan = planner::build_plan(body).map_err(|e| worker::Error::RustError(e.to_string()))?;
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
