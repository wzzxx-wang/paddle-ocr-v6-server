use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use image::{DynamicImage, ImageFormat};
use ocr_rs::{Backend, OcrEngine, OcrEngineConfig, OcrResult_};
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

fn main() {}
