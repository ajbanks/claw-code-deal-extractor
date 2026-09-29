//! Router-level tests: exercise the axum app without any LLM calls.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

// The router is built in main.rs, which tests can't import — rebuild the
// same routes against the lib functions instead. That keeps the test a
// real HTTP round-trip without duplicating handler logic.
fn app() -> axum::Router {
    use std::sync::Arc;
    axum::Router::new()
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/v1/validate",
            axum::routing::post(
                |axum::Json(req): axum::Json<deal_extractor_service::ValidateRequest>| async move {
                    axum::Json(deal_extractor_service::run_validation(&req))
                },
            ),
        )
        .with_state(Arc::new(()))
}

#[tokio::test]
async fn health() {
    let resp = app()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn validate_accepts_and_reports() {
    let body = serde_json::json!({
        "document": {
            "canonical_deal": {
                "facilities": [],
                "focus_facility_id": "missing",
            }
        }
    });
    let resp = app()
        .oneshot(
            Request::post("/v1/validate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1_000_000)
        .await
        .unwrap();
    let out: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(out["valid"], false);
    assert!(!out["issues"].as_array().unwrap().is_empty());
}
