//! `deal-extractor-service` — HTTP API for property-deal extraction.
//!
//! The TS backend sends extracted document pages; the service runs
//! whole-corpus LLM extraction, validates against the shared JSON Schema +
//! domain invariants, and drives a tool-assisted repair loop. The agent
//! owns correctness — no deterministic coercion of values.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use deal_extractor_service::{
    ErrorBody, ExtractRequest, ExtractResponse, ValidateRequest, ValidateResponse,
};
use tower_http::{limit::RequestBodyLimitLayer, timeout::TimeoutLayer};

#[derive(Parser)]
#[command(
    name = "deal-extractor-service",
    about = "Property-deal extraction HTTP service"
)]
struct Cli {
    /// Listen port.
    #[arg(long, env = "DEAL_EXTRACTOR_PORT", default_value = "8090")]
    port: u16,
    /// Listen host.
    #[arg(long, env = "DEAL_EXTRACTOR_HOST", default_value = "0.0.0.0")]
    host: String,
}

#[derive(Clone)]
struct AppState;

fn app_router() -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/schema", get(schema))
        .route("/v1/validate", post(validate))
        .route("/v1/extract", post(extract))
        // 200 MiB body cap — enough for base64-encoded image-heavy PDFs.
        .layer(DefaultBodyLimit::max(200 * 1024 * 1024))
        .layer(RequestBodyLimitLayer::new(200 * 1024 * 1024))
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(600),
        ))
        .with_state(Arc::new(AppState))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let addr = format!("{}:{}", cli.host, cli.port);
    let app = app_router();
    eprintln!("deal-extractor-service listen=http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn schema() -> Json<serde_json::Value> {
    Json(deal_extractor::contract::profile_schema())
}

async fn validate(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ValidateRequest>,
) -> Json<ValidateResponse> {
    Json(deal_extractor_service::run_validation(&req))
}

async fn extract(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>, (StatusCode, Json<ErrorBody>)> {
    deal_extractor_service::run_extraction(&req)
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(ErrorBody { error: e }),
            )
        })
}
