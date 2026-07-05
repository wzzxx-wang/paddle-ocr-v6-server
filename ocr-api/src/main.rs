use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use image::DynamicImage;
use multer::Multipart;
use ocr_rs::{OcrEngine, OcrEngineConfig, OcrResult_};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tokio::sync::Semaphore;

/// Configuration loaded from environment variables
#[derive(Debug, Clone)]
struct AppConfig {
    host: String,
    port: u16,
    det_model: String,
    rec_model: String,
    charset: String,
    threads: i32,
    concurrency: usize,
    confidence: f32,
    max_image_size: u32,
    max_payload_size: usize,
}

impl AppConfig {
    fn from_env() -> Self {
        fn env_or(key: &str, default: &str) -> String {
            std::env::var(key).unwrap_or_else(|_| default.to_string())
        }
        fn env_or_parse<T: std::str::FromStr>(key: &str, default: &str) -> T {
            env_or(key, default)
                .parse()
                .unwrap_or_else(|_| panic!("Invalid value for {}: expected number", key))
        }

        Self {
            host: env_or("OCR_HOST", "0.0.0.0"),
            port: env_or_parse("OCR_PORT", "8080"),
            det_model: env_or("OCR_DET_MODEL", "models/PP-OCRv6_medium_det.mnn"),
            rec_model: env_or("OCR_REC_MODEL", "models/PP-OCRv6_medium_rec.mnn"),
            charset: env_or("OCR_CHARSET", "models/ppocr_keys_v6_medium.txt"),
            threads: env_or_parse("OCR_THREADS", "4"),
            concurrency: env_or_parse("OCR_CONCURRENCY", "10"),
            confidence: env_or_parse("OCR_CONFIDENCE", "0.5"),
            max_image_size: env_or_parse("OCR_MAX_IMAGE_SIZE", "4096"),
            max_payload_size: env_or_parse("OCR_MAX_PAYLOAD_SIZE", "20971520"),
        }
    }
}

/// Shared application state
struct AppState {
    engine: Arc<OcrEngine>,
    semaphore: Semaphore,
    start_time: Instant,
    max_image_size: u32,
    max_payload_size: usize,
}

/// Unified API error — auto-maps to JSON error responses
#[derive(Debug, Error)]
enum AppError {
    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Payload too large")]
    PayloadTooLarge,

    #[error("OCR processing failed: {0}")]
    OcrError(#[from] ocr_rs::OcrError),

    #[error("Image decode failed: {0}")]
    ImageError(#[from] image::ImageError),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_msg) = match &self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::PayloadTooLarge => {
                (StatusCode::PAYLOAD_TOO_LARGE, "Payload too large".to_string())
            }
            AppError::OcrError(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            AppError::ImageError(e) => {
                (StatusCode::BAD_REQUEST, format!("Image decode failed: {}", e))
            }
        };

        let body = serde_json::json!({
            "success": false,
            "error": error_msg,
        });
        (status, Json(body)).into_response()
    }
}

// ----- Response DTOs -----

#[derive(Debug, Clone, Serialize)]
struct OcrItem {
    text: String,
    confidence: f32,
    bbox: Bbox,
}

#[derive(Debug, Clone, Serialize)]
struct Bbox {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
}

impl From<OcrResult_> for OcrItem {
    fn from(r: OcrResult_) -> Self {
        OcrItem {
            text: r.text,
            confidence: r.confidence,
            bbox: Bbox {
                left: r.bbox.rect.left(),
                top: r.bbox.rect.top(),
                width: r.bbox.rect.width(),
                height: r.bbox.rect.height(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct ApiResponse<T: Serialize> {
    success: bool,
    data: T,
}

impl<T: Serialize> ApiResponse<T> {
    fn ok(data: T) -> Json<Self> {
        Json(Self {
            success: true,
            data,
        })
    }
}

// ----- Request DTOs -----

#[derive(Debug, Deserialize)]
struct OcrJsonRequest {
    image: String,
}

#[derive(Debug, Deserialize)]
struct OcrBatchJsonRequest {
    images: Vec<String>,
}

// ----- Image helpers -----

/// Decode image bytes → DynamicImage, scaling down if too large
fn decode_image(bytes: &[u8], max_size: u32) -> Result<DynamicImage, AppError> {
    let img = image::load_from_memory(bytes).map_err(AppError::ImageError)?;
    resize_if_needed(img, max_size)
}

fn resize_if_needed(img: DynamicImage, max_size: u32) -> Result<DynamicImage, AppError> {
    let (w, h) = (img.width(), img.height());
    if w <= max_size && h <= max_size {
        return Ok(img);
    }
    let ratio = (max_size as f64) / (w.max(h) as f64);
    let new_w = (w as f64 * ratio) as u32;
    let new_h = (h as f64 * ratio) as u32;
    Ok(img.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3))
}

// ----- Multipart decode helper -----

/// Parse a multipart form body to extract the "file" field as raw bytes.
/// Returns Err(BadRequest) if no file field is found.
async fn decode_multipart_body(
    body: &[u8],
    content_type: &str,
) -> Result<Vec<u8>, AppError> {
    // Extract boundary from Content-Type header
    let boundary = content_type
        .split("boundary=")
        .nth(1)
        .and_then(|s| {
            let s = s.trim();
            // strip optional quotes
            Some(s.trim_matches('"').to_string())
        })
        .ok_or_else(|| AppError::BadRequest("Missing multipart boundary".to_string()))?;

    use std::convert::Infallible;
    let stream = tokio_stream::once(Ok::<Bytes, Infallible>(Bytes::copy_from_slice(body)));
    let mut multipart = Multipart::new(stream, boundary);

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::BadRequest(format!("Multipart parse error: {}", e)))?
    {
        if field.name() == Some("file") {
            let data = field
                .bytes()
                .await
                .map_err(|e| AppError::BadRequest(format!("File field read error: {}", e)))?;
            return Ok(data.to_vec());
        }
    }

    Err(AppError::BadRequest(
        "No 'file' field found in multipart form".to_string(),
    ))
}

// ----- JSON base64 decode helpers -----

/// Parse JSON body to extract base64-encoded image
fn decode_json_body(body: &[u8]) -> Result<DynamicImage, AppError> {
    let req: OcrJsonRequest =
        serde_json::from_slice(body).map_err(|e| AppError::BadRequest(format!("Invalid JSON: {}", e)))?;
    let bytes = base64_decode(&req.image)?;
    decode_image(&bytes, u32::MAX) // caller will resize
}

fn base64_decode(encoded: &str) -> Result<Vec<u8>, AppError> {
    // Strip optional data URI prefix: "data:image/png;base64,..."
    let stripped = if let Some(pos) = encoded.find("base64,") {
        &encoded[pos + 7..]
    } else {
        encoded
    };
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(stripped)
        .map_err(|e| AppError::BadRequest(format!("Base64 decode error: {}", e)))
}

// ----- Handler functions -----

#[derive(Debug, Clone, Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    model: String,
    backend: String,
    uptime_secs: u64,
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let backend = if cfg!(feature = "cuda") {
        "cuda"
    } else {
        "cpu"
    };

    Json(HealthResponse {
        status: "ok".to_string(),
        version: "2.3.1".to_string(),
        model: "PP-OCRv6_medium".to_string(),
        backend: backend.to_string(),
        uptime_secs: state.start_time.elapsed().as_secs(),
    })
}

async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    req: axum::http::Request<axum::body::Body>,
) -> Result<Json<ApiResponse<Vec<OcrItem>>>, AppError> {
    let _permit = state.semaphore.acquire().await;
    let start = Instant::now();

    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = axum::body::to_bytes(req.into_body(), state.max_payload_size)
        .await
        .map_err(|_| AppError::PayloadTooLarge)?;

    let img = if content_type.starts_with("multipart/form-data") {
        let raw = decode_multipart_body(&body, &content_type).await?;
        decode_image(&raw, state.max_image_size)?
    } else if content_type.starts_with("application/json") {
        let mut img = decode_json_body(&body)?;
        img = resize_if_needed(img, state.max_image_size)?;
        img
    } else {
        return Err(AppError::BadRequest(format!(
            "Unsupported Content-Type: {}",
            content_type
        )));
    };

    let results = state.engine.recognize(&img)?;

    log::info!(
        "POST /ocr - 200 OK - {} text regions - {:.3}s",
        results.len(),
        start.elapsed().as_secs_f64()
    );

    let items: Vec<OcrItem> = results.into_iter().map(OcrItem::from).collect();
    Ok(ApiResponse::ok(items))
}

async fn ocr_batch_handler(
    State(state): State<Arc<AppState>>,
    req: axum::http::Request<axum::body::Body>,
) -> Result<Json<ApiResponse<Vec<Vec<OcrItem>>>>, AppError> {
    let _permit = state.semaphore.acquire().await;
    let start = Instant::now();

    let body = axum::body::to_bytes(req.into_body(), state.max_payload_size)
        .await
        .map_err(|_| AppError::PayloadTooLarge)?;

    let batch: OcrBatchJsonRequest = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {}", e)))?;

    let mut all_results: Vec<Vec<OcrItem>> = Vec::with_capacity(batch.images.len());

    for encoded in &batch.images {
        let raw = base64_decode(encoded)?;
        let img = decode_image(&raw, state.max_image_size)?;
        let results = state.engine.recognize(&img)?;
        all_results.push(results.into_iter().map(OcrItem::from).collect());
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_regions: usize = all_results.iter().map(|r| r.len()).sum();
    log::info!(
        "POST /ocr/batch - 200 OK - {} images, {} text regions total - {:.3}s",
        batch.images.len(),
        total_regions,
        elapsed,
    );

    Ok(ApiResponse::ok(all_results))
}

fn main() {}
