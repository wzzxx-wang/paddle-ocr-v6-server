# OCR API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained HTTP API service wrapping `ocr-rs` (rust-paddle-ocr v2.3.1) with PP-OCRv6 medium, deployable via Docker (CPU/GPU).

**Architecture:** Axum HTTP server with singleton `Arc<OcrEngine>` injected via `AppState`. Concurrency via `Semaphore`. Build-time feature selects `Backend::CPU` or `Backend::CUDA`. All config via env vars.

**Tech Stack:** Rust, Axum 0.7, Tokio, ocr-rs 2.3.1 (git dep), MNN, Docker multi-stage build.

## Global Constraints

- Rust edition 2021
- `ocr-rs` git dep: `{ git = "https://github.com/zibo-chen/rust-paddle-ocr.git", tag = "v2.3.1", features = ["build-mnn-from-source"] }`
- Axum 0.7, Tokio 1.x (full), image 0.25, serde_json, base64 0.22, multer 3.x
- Features: `cpu` (default), `cuda` (enables `ocr-rs/cuda`)
- Models locally cached in `models/` (gitignored), COPY'd into Docker image
- All config via env vars with defaults (spec §5)
- Single binary, no lazy loading — engine init before server starts
- GPU hardcodes `Backend::CUDA` via `#[cfg(feature = "cuda")]`
- `OcrEngineConfig.min_result_confidence` ← `OCR_CONFIDENCE` env var

---

## File Structure

```
/data/ai_claude/ocr_pp_v6_api/ocr-api/
├── Cargo.toml
├── .gitignore
├── src/
│   └── main.rs              # AppState, handlers, helpers, error types — entire API
├── download-models.sh
├── Dockerfile
├── Dockerfile.gpu
└── README.md
```

All code in `src/main.rs` — it's a thin HTTP glue layer over `ocr-rs`.

---

### Task 1: Project Scaffold

**Files:**
- Create: `Cargo.toml`
- Create: `.gitignore`

- [ ] **Step 1: Create `Cargo.toml`**

```toml
[package]
name = "ocr-api"
version = "0.1.0"
edition = "2021"

[features]
default = ["cpu"]
cpu = []
cuda = ["ocr-rs/cuda"]

[dependencies]
ocr-rs = { git = "https://github.com/zibo-chen/rust-paddle-ocr.git", tag = "v2.3.1", features = ["build-mnn-from-source"] }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
image = "0.25"
env_logger = "0.11"
log = "0.4"
thiserror = "1"
base64 = "0.22"
multer = "3"
uuid = { version = "1", features = ["v4"] }
```

- [ ] **Step 2: Create `.gitignore`**

```
/target
models/
```

- [ ] **Step 3: Create `src/main.rs` placeholder**

```rust
fn main() {
    println!("ocr-api placeholder");
}
```

- [ ] **Step 4: Verify compilation**

```bash
cargo check
```
Expected: `Checking ocr-api ... Finished`. First run downloads ocr-rs from git + builds MNN from source (~5-10 min).

---

### Task 2: Model Download Script

**Files:**
- Create: `download-models.sh`

- [ ] **Step 1: Create download script**

```bash
#!/bin/bash
set -e

REPO="https://github.com/zibo-chen/rust-paddle-ocr"
TAG="v2.3.1"
MODELS_DIR="models"

mkdir -p "$MODELS_DIR"

echo "Downloading PP-OCRv6 medium detection model..."
curl -# -L -o "$MODELS_DIR/PP-OCRv6_medium_det.mnn" \
  "$REPO/raw/$TAG/models/PP-OCRv6_medium_det.mnn"

echo "Downloading PP-OCRv6 medium recognition model..."
curl -# -L -o "$MODELS_DIR/PP-OCRv6_medium_rec.mnn" \
  "$REPO/raw/$TAG/models/PP-OCRv6_medium_rec.mnn"

echo "Downloading PP-OCRv6 medium charset..."
curl -# -L -o "$MODELS_DIR/ppocr_keys_v6_medium.txt" \
  "$REPO/raw/$TAG/models/ppocr_keys_v6_medium.txt"

echo "Done. Files saved to $MODELS_DIR/"
ls -lh "$MODELS_DIR/"
```

- [ ] **Step 2: Make executable and run**

```bash
chmod +x download-models.sh
./download-models.sh
```

Expected: three files in `models/`, ~67 MB total.

---

### Task 3: Core Module — Config, State, Error, DTOs, Helpers

**Files:**
- Modify: `src/main.rs` (replace placeholder with full module)

- [ ] **Step 1: Add imports, `AppConfig`, `AppState`, and `AppError`**

Replace the placeholder `fn main()` with:

```rust
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
```

- [ ] **Step 2: Add response types and image helpers (append after AppError)**

```rust
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
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check 2>&1 | head -20
```
Expected: compiles with type-checked code (may warn about unused functions — that's fine).

---

### Task 4: Request Decode Helpers — Multipart & JSON

**Files:**
- Modify: `src/main.rs` (append helper functions)

- [ ] **Step 1: Add multipart decode helper**

```rust
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

    let mut multipart = Multipart::new(body, boundary.as_str());
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
```

- [ ] **Step 2: Add JSON base64 decode helpers**

```rust
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
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check 2>&1 | head -20
```
Expected: compiles.

---

### Task 5: Handlers — ocr, batch, health

**Files:**
- Modify: `src/main.rs` (append handler functions)

- [ ] **Step 1: Add `health_handler()`**

```rust
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
```

- [ ] **Step 2: Add `ocr_handler()` — supports multipart and JSON**

```rust
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
```

- [ ] **Step 3: Add `ocr_batch_handler()` — JSON-only, serial processing**

```rust
async fn ocr_batch_handler(
    State(state): State<Arc<AppState>>,
    content_type: axom::http::HeaderMap,
    body: axom::body::Bytes,
) -> Result<Json<ApiResponse<Vec<Vec<OcrItem>>>>, AppError> {
    // JSON only
    let _permit = state.semaphore.acquire().await;
    let start = Instant::now();

    let req: OcrBatchJsonRequest = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {}", e)))?;

    let mut all_results: Vec<Vec<OcrItem>> = Vec::with_capacity(req.images.len());

    for encoded in &req.images {
        let raw = base64_decode(encoded)?;
        let img = decode_image(&raw, state.max_image_size)?;
        let results = state.engine.recognize(&img)?;
        all_results.push(results.into_iter().map(OcrItem::from).collect());
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_regions: usize = all_results.iter().map(|r| r.len()).sum();
    log::info!(
        "POST /ocr/batch - 200 OK - {} images, {} text regions total - {:.3}s",
        req.images.len(),
        total_regions,
        elapsed,
    );

    Ok(ApiResponse::ok(all_results))
}
```

Wait — I have a typo: `axom` should be `axum`. And I need to import `axum::body::Bytes`. Let me fix this.

Actually, let me reconsider the handler signatures. In Axum 0.7, extracting both `State` and raw `Request` at the same time can be tricky. Let me use a cleaner approach:

```rust
async fn ocr_batch_handler(
    State(state): State<Arc<AppState>>,
    req: axum::http::Request<axum::body::Body>,
) -> Result<Json<ApiResponse<Vec<Vec<OcrItem>>>>, AppError> {
```

This is consistent with `ocr_handler`.

- [ ] **Step 3 (fixed): Add `ocr_batch_handler()`**

```rust
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
```

- [ ] **Step 4: Verify compilation**

```bash
cargo check 2>&1 | head -30
```
Expected: compiles (may still warn about unused items in test-only code).

---

### Task 6: `main()` — Engine Init, Router, Server Start

**Files:**
- Modify: `src/main.rs` (replace placeholder `main()` with real one)

- [ ] **Step 1: Replace placeholder `fn main()` with full implementation**

```rust
#[tokio::main]
async fn main() {
    env_logger::init();

    log::info!("Loading configuration...");
    let config = AppConfig::from_env();

    // Determine backend at compile time
    #[cfg(feature = "cuda")]
    let backend = Backend::CUDA;
    #[cfg(not(feature = "cuda"))]
    let backend = Backend::CPU;

    log::info!(
        "Initializing OCR engine (backend: {:?}, threads: {})...",
        backend,
        config.threads,
    );

    let engine_config = OcrEngineConfig::new()
        .with_backend(backend)
        .with_threads(config.threads)
        .with_min_result_confidence(config.confidence);

    let engine = OcrEngine::new(
        &config.det_model,
        &config.rec_model,
        &config.charset,
        Some(engine_config),
    )
    .expect("Failed to initialize OCR engine — check model files");

    log::info!("OCR engine ready. Starting server...");

    let state = Arc::new(AppState {
        engine: Arc::new(engine),
        semaphore: Semaphore::new(config.concurrency),
        start_time: Instant::now(),
        max_image_size: config.max_image_size,
        max_payload_size: config.max_payload_size,
    });

    let app = Router::new()
        .route("/ocr", post(ocr_handler))
        .route("/ocr/batch", post(ocr_batch_handler))
        .route("/health", get(health_handler))
        .with_state(state);

    let addr = format!("{}:{}", config.host, config.port);
    log::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind address");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
```

- [ ] **Step 2: Verify full compilation**

```bash
cargo check 2>&1
```
Expected: compiles without errors.

- [ ] **Step 3: Run a quick sanity check (will fail without models — that's expected)**

```bash
cargo run 2>&1 | head -5
```
Expected: errors because `models/` doesn't exist or models not found. That's the correct behavior — models must be downloaded first.

---

### Task 7: Dockerfile (CPU)

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM rust:1-slim-bookworm AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y cmake pkg-config && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml ./
COPY src/ ./src/
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libgomp1 libatomic1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/ocr-api /usr/local/bin/ocr-api
COPY models/ /app/models/
EXPOSE 8080
CMD ["ocr-api"]
```

- [ ] **Step 2: Build the CPU Docker image**

```bash
docker build -t ocr-api:cpu .
```
Expected: builds successfully (~15-20 min first time due to MNN source build).

---

### Task 8: Dockerfile.gpu (CUDA)

**Files:**
- Create: `Dockerfile.gpu`

- [ ] **Step 1: Create `Dockerfile.gpu`**

```dockerfile
FROM nvidia/cuda:12.4-devel-bookworm AS builder
WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y cmake pkg-config && rm -rf /var/lib/apt/lists/*

# Pre-fetch git dep source so cargo can build
COPY Cargo.toml ./
COPY src/ ./src/
RUN cargo build --release --features cuda

FROM nvidia/cuda:12.4-runtime-bookworm
RUN apt-get update && apt-get install -y libgomp1 libatomic1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/ocr-api /usr/local/bin/ocr-api
COPY models/ /app/models/
EXPOSE 8080
CMD ["ocr-api"]
```

- [ ] **Step 2: Build the GPU Docker image**

```bash
docker build -t ocr-api:gpu -f Dockerfile.gpu .
```
Expected: builds successfully. Requires `nvidia-container-toolkit` for the runtime.

---

### Task 9: Tests

**Files:**
- Modify: `src/main.rs` (append `#[cfg(test)] mod tests`)

- [ ] **Step 1: Add unit tests for helpers and response types**

Append to `src/main.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_if_needed_within_limit() {
        let img = DynamicImage::new_rgb8(800, 600);
        let result = resize_if_needed(img, 4096).unwrap();
        assert_eq!(result.width(), 800);
        assert_eq!(result.height(), 600);
    }

    #[test]
    fn test_resize_if_needed_exceeds_limit() {
        let img = DynamicImage::new_rgb8(8000, 6000);
        let result = resize_if_needed(img, 4096).unwrap();
        assert!(result.width() <= 4096);
        assert!(result.height() <= 4096);
        // Aspect ratio should be preserved (8000:6000 = 4:3)
        assert!((result.width() as f64 / result.height() as f64 - 4.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_ocr_item_from_result_round_trip() {
        let bbox = ocr_rs::TextBox::new(
            imageproc::rect::Rect::at(10, 20).of_size(200, 30),
            0.9,
        );
        let result = OcrResult_::new("hello".into(), 0.95, bbox);
        let item: OcrItem = result.into();
        assert_eq!(item.text, "hello");
        assert!((item.confidence - 0.95).abs() < 0.001);
        assert_eq!(item.bbox.left, 10);
        assert_eq!(item.bbox.top, 20);
        assert_eq!(item.bbox.width, 200);
        assert_eq!(item.bbox.height, 30);
    }

    #[test]
    fn test_base64_decode_round_trip() {
        let original = b"hello world";
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(original);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_base64_decode_with_data_uri() {
        let encoded = "data:image/png;base64,aGVsbG8=";
        let decoded = base64_decode(encoded).unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn test_base64_decode_invalid() {
        let result = base64_decode("!!!not-valid-base64!!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_app_error_bad_request_response() {
        let err = AppError::BadRequest("test error".into());
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_app_error_payload_too_large_response() {
        let err = AppError::PayloadTooLarge;
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test
```
Expected: all tests pass.

---

### Task 10: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# OCR API

基于 [rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) (PP-OCRv6 medium) 的 HTTP API 服务，支持 Docker 单机部署。

## Quick Start

```bash
# 1. 下载模型（仅首次）
./download-models.sh

# 2. 构建并运行（CPU）
docker build -t ocr-api:cpu .
docker run -d -p 8080:8080 ocr-api:cpu
```

## API

### `POST /ocr`

支持 multipart/form-data 或 JSON base64：

```bash
# Multipart
curl -F "file=@test.jpg" http://localhost:8080/ocr

# JSON base64
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w0 test.jpg)'"}'
```

### `POST /ocr/batch`

```bash
curl -X POST http://localhost:8080/ocr/batch \
  -H "Content-Type: application/json" \
  -d '{"images": ["<base64>", "<base64>"]}'
```

### `GET /health`

```bash
curl http://localhost:8080/health
```

## Configuration

所有配置通过环境变量注入，参见 [spec](docs/superpowers/specs/2026-07-05-ocr-api-design.md)。

## GPU 版本

```bash
docker build -t ocr-api:gpu -f Dockerfile.gpu .
docker run -d -p 8080:8080 --gpus all ocr-api:gpu
```
```

---

## Spec Coverage Check

| Spec § | Requirement | Task |
|---|---|---|
| §1 Architecture | Axum + Tokio + Arc\<OcrEngine\> | Task 6 |
| §1 Concurrency | Semaphore | Task 6 |
| §1 GPU strategy | Cargo feature, compile-time | Task 6 |
| §2.1 POST /ocr | multipart + JSON | Task 5 Step 2 |
| §2.1 Response format | text, confidence, bbox | Task 3 Step 2 |
| §2.2 POST /ocr/batch | JSON, serial processing | Task 5 Step 3 |
| §2.3 GET /health | status, version, model, backend, uptime | Task 5 Step 1 |
| §2.4 Error response | unified {success,error} | Task 3 Step 1 |
| §3 Project structure | Cargo.toml, main.rs, models/, Dockerfiles | Tasks 1, 7, 8 |
| §4 Cargo.toml dependencies | git dep on ocr-rs, features | Task 1 |
| §5 Configuration | env vars with defaults | Task 3 Step 1 |
| §6 Logging | env_logger, info log lines | Tasks 5, 6 |
| §7 Models | download-models.sh, Docker COPY | Task 2 |
| §8 Backend choice | #[cfg] compile-time | Task 6 |
| §9 Docker (CPU) | multi-stage, slim runtime | Task 7 |
| §9 Docker (GPU) | CUDA 12.4, multi-stage | Task 8 |
| §10 Request protection | payload limit, image size, semaphore | Tasks 3, 5 |
| §11 Testing | unit tests for helpers, error types | Task 9 |
