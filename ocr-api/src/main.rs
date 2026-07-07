use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{get, post},
    Router,
};
use bytes::Bytes;
use image::DynamicImage;
use multer::Multipart;
use ocr_rs::{Backend, OcrEngine, OcrEngineConfig, OcrResult_, TextBox, RecognitionResult};
use serde::{Deserialize, Serialize};
use lru::LruCache;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use thiserror::Error;
use tokio::sync::Semaphore;

/// Configuration loaded from environment variables
#[derive(Debug, Clone)]
struct AppConfig {
    host: String,
    port: u16,
    threads: i32,
    concurrency: usize,
    confidence: f32,
    infer_concurrency: usize,
    cache_max_mb: usize,
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
            threads: env_or_parse("OCR_THREADS", "4"),
            concurrency: env_or_parse("OCR_CONCURRENCY", "10"),
            confidence: env_or_parse("OCR_CONFIDENCE", "0.5"),
            infer_concurrency: env_or_parse("OCR_INFER_CONCURRENCY", "2"),
            cache_max_mb: env_or_parse("OCR_CACHE_MAX_MB", "500"),
            max_image_size: env_or_parse("OCR_MAX_IMAGE_SIZE", "4096"),
            max_payload_size: env_or_parse("OCR_MAX_PAYLOAD_SIZE", "20971520"),
        }
    }
}

/// Model variants available for runtime switching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ModelVariant {
    Medium,
    Small,
    Tiny,
}

impl ModelVariant {
    fn all() -> [(&'static str, ModelVariant); 3] {
        [
            ("medium", ModelVariant::Medium),
            ("small", ModelVariant::Small),
            ("tiny", ModelVariant::Tiny),
        ]
    }

    fn det_path(self) -> &'static str {
        match self {
            ModelVariant::Medium => "models/PP-OCRv6_medium_det.mnn",
            ModelVariant::Small => "models/PP-OCRv6_small_det.mnn",
            ModelVariant::Tiny => "models/PP-OCRv6_tiny_det.mnn",
        }
    }

    fn rec_path(self) -> &'static str {
        match self {
            ModelVariant::Medium => "models/PP-OCRv6_medium_rec.mnn",
            ModelVariant::Small => "models/PP-OCRv6_small_rec.mnn",
            ModelVariant::Tiny => "models/PP-OCRv6_tiny_rec.mnn",
        }
    }

    fn charset_path(self) -> &'static str {
        match self {
            ModelVariant::Medium => "models/ppocr_keys_v6_medium.txt",
            ModelVariant::Small => "models/ppocr_keys_v6_small.txt",
            ModelVariant::Tiny => "models/ppocr_keys_v6_tiny.txt",
        }
    }

}

/// Pre-loaded engines for all model variants
struct ModelSet {
    engines: HashMap<String, Arc<OcrEngine>>,
}

impl ModelSet {
    fn load(config: &AppConfig) -> Result<Self, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        let backend = Backend::CUDA;
        #[cfg(not(feature = "cuda"))]
        let backend = Backend::CPU;

        let mut engines = HashMap::new();
        for (name, variant) in ModelVariant::all() {
            log::info!("Loading {name} model...");
            let engine = OcrEngine::new(
                variant.det_path(),
                variant.rec_path(),
                variant.charset_path(),
                Some(
                    OcrEngineConfig::new()
                        .with_backend(backend)
                        .with_threads(config.threads)
                        .with_min_result_confidence(config.confidence),
                ),
            )?;
            engines.insert(name.to_string(), Arc::new(engine));
        }

        Ok(ModelSet { engines })
    }

    fn get(&self, name: &str) -> Option<Arc<OcrEngine>> {
        self.engines.get(name).cloned()
    }
}

/// Shared application state
struct AppState {
    models: ModelSet,
    semaphore: Semaphore,
    infer_semaphore: Arc<tokio::sync::Semaphore>,
    min_confidence: f32,
    cache: OcrCache,
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
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OcrBatchJsonRequest {
    images: Vec<String>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OcrQueryParams {
    model: Option<String>,
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

// ----- LRU Cache for OCR results -----

/// Approximate byte size of a Vec<OcrItem> (used for cache eviction accounting).
fn estimate_items_size(items: &[OcrItem]) -> usize {
    items.len() * std::mem::size_of::<OcrItem>()
        + items.iter().map(|i| i.text.capacity()).sum::<usize>()
}

/// Compute a cache key from raw image bytes + model name.
fn hash_raw(bytes: &[u8], model: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    model.hash(&mut hasher);
    hasher.finish()
}

/// Size-aware LRU cache. Evicts oldest entries when total exceeds max_bytes.
struct OcrCache {
    inner: Mutex<OcrCacheInner>,
    max_bytes: usize,
}

struct OcrCacheInner {
    lru: LruCache<u64, Vec<OcrItem>>,
    current_bytes: usize,
}

impl OcrCache {
    fn new(max_bytes: usize) -> Self {
        // LruCache entry-count cap — set high so byte-size eviction governs instead.
        let count_cap = NonZeroUsize::new(1_000_000).expect("1M is non-zero");
        Self {
            inner: Mutex::new(OcrCacheInner {
                lru: LruCache::new(count_cap),
                current_bytes: 0,
            }),
            max_bytes,
        }
    }

    fn get(&self, key: &u64) -> Option<Vec<OcrItem>> {
        let mut inner = self.inner.lock().expect("cache lock poisoned");
        inner.lru.get(key).cloned()
    }

    fn put(&self, key: u64, items: Vec<OcrItem>) {
        let entry_size = estimate_items_size(&items);
        let mut inner = self.inner.lock().expect("cache lock poisoned");

        // If key already exists, remove its size from accounting first.
        let old_size = inner.lru.peek(&key).map(|v| estimate_items_size(v)).unwrap_or(0);
        inner.current_bytes = inner.current_bytes.saturating_sub(old_size);

        // Evict oldest entries until there's room (or cache is empty).
        while inner.current_bytes + entry_size > self.max_bytes && inner.lru.len() > 0 {
            if let Some((_, evicted)) = inner.lru.pop_lru() {
                inner.current_bytes =
                    inner.current_bytes.saturating_sub(estimate_items_size(&evicted));
            }
        }

        inner.lru.put(key, items);
        inner.current_bytes += entry_size;
    }
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

/// Run detection then concurrently recognize each text line.
/// Results sorted top-to-bottom, left-to-right, filtered by recognition confidence.
async fn recognize_text_lines(
    engine: Arc<OcrEngine>,
    img: DynamicImage,
    state: &AppState,
) -> Result<Vec<OcrItem>, AppError> {
    // Step 1: Detect text regions (CPU-bound via spawn_blocking)
    let det_engine = engine.clone();
    let img_for_det = img.clone();
    let text_boxes: Vec<TextBox> = tokio::task::spawn_blocking(move || {
        det_engine.detect(&img_for_det)
    })
    .await
    .map_err(|e| AppError::BadRequest(format!("Detection join error: {}", e)))?
    .map_err(AppError::OcrError)?;

    if text_boxes.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Crop each text region and spawn concurrent recognition tasks
    let mut handles = Vec::with_capacity(text_boxes.len());

    for tb in text_boxes {
        let e = engine.clone();
        let rect = tb.rect;
        let crop = img.crop_imm(rect.left().max(0) as u32, rect.top().max(0) as u32, rect.width() as u32, rect.height() as u32);
        // Acquire permit before spawning (async context) — gates concurrency
        let permit = state.infer_semaphore
            .clone()
            .acquire_owned()
            .await
            .expect("infer semaphore closed");

        handles.push(tokio::task::spawn_blocking(move || {
            let _permit = permit; // held for duration of recognize
            let rec = e.recognize_text(&crop)?;
            Ok::<(TextBox, RecognitionResult), AppError>((tb, rec))
        }));
    }

    // Step 3: Collect results, filter by confidence, format into OcrItem
    let mut items: Vec<OcrItem> = Vec::with_capacity(handles.len());
    for handle in handles {
        let (tb, rec) = handle
            .await
            .map_err(|e| AppError::BadRequest(format!("Recognize join error: {}", e)))??;

        if rec.confidence < state.min_confidence {
            continue;
        }

        items.push(OcrItem {
            text: rec.text,
            confidence: rec.confidence,
            bbox: Bbox {
                left: tb.rect.left(),
                top: tb.rect.top(),
                width: tb.rect.width() as u32,
                height: tb.rect.height() as u32,
            },
        });
    }

    // Step 4: Sort top-to-bottom, left-to-right
    items.sort_by_key(|item| (item.bbox.top, item.bbox.left));

    Ok(items)
}

// ----- JSON base64 decode helpers -----

/// Parse JSON body to extract base64-encoded image and optional model variant
fn decode_json_body(body: &[u8]) -> Result<OcrJsonRequest, AppError> {
    serde_json::from_slice::<OcrJsonRequest>(body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {}", e)))
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
    Query(params): Query<OcrQueryParams>,
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

    // Determine model variant: query param -> JSON field -> default "medium"
    let (raw_bytes, model_name) = if content_type.starts_with("multipart/form-data") {
        let raw = decode_multipart_body(&body, &content_type).await?;
        let model_name = params.model.clone().unwrap_or_else(|| "medium".to_string());
        (raw, model_name)
    } else if content_type.starts_with("application/json") {
        let req = decode_json_body(&body)?;
        let bytes = base64_decode(&req.image)?;
        let model_name = req.model
            .or_else(|| params.model.clone())
            .unwrap_or_else(|| "medium".to_string());
        (bytes, model_name)
    } else {
        return Err(AppError::BadRequest(format!(
            "Unsupported Content-Type: {}",
            content_type
        )));
    };

    // Check cache before OCR
    let cache_key = hash_raw(&raw_bytes, &model_name);
    if let Some(cached) = state.cache.get(&cache_key) {
        log::info!(
            "POST /ocr - 200 OK - {} text regions (cached) - model: {} - {:.3}s",
            cached.len(),
            model_name,
            start.elapsed().as_secs_f64()
        );
        return Ok(ApiResponse::ok(cached));
    }

    let img = decode_image(&raw_bytes, state.max_image_size)?;

    let engine = state.models.get(&model_name).ok_or_else(|| {
        AppError::BadRequest(format!(
            "Unknown model variant: {}. Supported: medium, small, tiny",
            model_name
        ))
    })?;

    let items = recognize_text_lines(engine, img, &state).await?;

    // Store in cache (clone for logging, original moves into cache)
    state.cache.put(cache_key, items.clone());

    log::info!(
        "POST /ocr - 200 OK - {} text regions - model: {} - {:.3}s",
        items.len(),
        model_name,
        start.elapsed().as_secs_f64()
    );

    Ok(ApiResponse::ok(items))
}

async fn ocr_batch_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<OcrQueryParams>,
    req: axum::http::Request<axum::body::Body>,
) -> Result<Json<ApiResponse<Vec<Vec<OcrItem>>>>, AppError> {
    let _permit = state.semaphore.acquire().await;
    let start = Instant::now();

    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !content_type.starts_with("application/json") {
        return Err(AppError::BadRequest(format!(
            "Unsupported Content-Type: {}",
            content_type
        )));
    }

    let body = axum::body::to_bytes(req.into_body(), state.max_payload_size)
        .await
        .map_err(|_| AppError::PayloadTooLarge)?;

    let batch: OcrBatchJsonRequest = serde_json::from_slice(&body)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {}", e)))?;

    let model_name = batch.model
        .or_else(|| params.model.clone())
        .unwrap_or_else(|| "medium".to_string());

    let engine = state.models.get(&model_name).ok_or_else(|| {
        AppError::BadRequest(format!(
            "Unknown model variant: {}. Supported: medium, small, tiny",
            model_name
        ))
    })?;

    // Decode all base64 images to raw bytes (needed for cache key)
    let raw_images: Vec<Vec<u8>> = batch
        .images
        .iter()
        .map(|encoded| base64_decode(encoded))
        .collect::<Result<Vec<_>, AppError>>()?;

    // Check cache for each image; only process uncached ones
    let mut all_results: Vec<Option<Vec<OcrItem>>> = vec![None; raw_images.len()];
    let mut to_process: Vec<(usize, Vec<u8>)> = Vec::new();

    for (i, raw) in raw_images.iter().enumerate() {
        let cache_key = hash_raw(raw, &model_name);
        if let Some(cached) = state.cache.get(&cache_key) {
            all_results[i] = Some(cached);
        } else {
            to_process.push((i, raw.clone()));
        }
    }

    // Decode only uncached images
    let decoded: Vec<DynamicImage> = to_process
        .iter()
        .map(|(_, raw)| decode_image(raw, state.max_image_size))
        .collect::<Result<Vec<_>, AppError>>()?;

    // Process uncached images concurrently
    let mut handles = Vec::with_capacity(decoded.len());
    for (idx, img) in decoded.into_iter().enumerate() {
        let e = engine.clone();
        let state_clone = state.clone();
        handles.push(tokio::spawn(async move {
            let items = recognize_text_lines(e, img, &state_clone).await?;
            Ok::<(usize, Vec<OcrItem>), AppError>((idx, items))
        }));
    }

    // Collect fresh results and store in cache
    for handle in handles {
        let (idx_in_to_process, items) = handle
            .await
            .map_err(|e| AppError::BadRequest(format!("Task join error: {}", e)))??;
        let (orig_idx, raw) = &to_process[idx_in_to_process];
        state.cache.put(hash_raw(raw, &model_name), items.clone());
        all_results[*orig_idx] = Some(items);
    }

    // All results are now populated
    let all_results: Vec<Vec<OcrItem>> = all_results.into_iter().map(|r| r.unwrap()).collect();

    let elapsed = start.elapsed().as_secs_f64();
    let total_regions: usize = all_results.iter().map(|r| r.len()).sum();
    log::info!(
        "POST /ocr/batch - 200 OK - {} images, {} text regions total - model: {} - {:.3}s",
        batch.images.len(),
        total_regions,
        model_name,
        elapsed,
    );

    Ok(ApiResponse::ok(all_results))
}

#[tokio::main]
async fn main() {
    env_logger::init();

    dotenvy::dotenv().ok();

    log::info!("Loading configuration...");
    let config = AppConfig::from_env();

    log::info!(
        "Loading all model variants (medium, small, tiny) with backend: {}, threads: {}...",
        if cfg!(feature = "cuda") { "cuda" } else { "cpu" },
        config.threads,
    );

    let models = ModelSet::load(&config)
        .expect("Failed to initialize OCR engines — check model files");

    log::info!("All models loaded. Starting server...");

    let state = Arc::new(AppState {
        models,
        semaphore: Semaphore::new(config.concurrency),
        infer_semaphore: Arc::new(tokio::sync::Semaphore::new(config.infer_concurrency)),
        min_confidence: config.confidence,
        cache: OcrCache::new(config.cache_max_mb * 1024 * 1024),
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
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Server error");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

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

    #[test]
    fn test_text_box_sorting() {
        use ocr_rs::TextBox;
        use imageproc::rect::Rect;

        let tb1 = TextBox::new(Rect::at(5, 30).of_size(100, 20), 0.9);
        let tb2 = TextBox::new(Rect::at(10, 10).of_size(80, 20), 0.9);
        let tb3 = TextBox::new(Rect::at(0, 20).of_size(60, 15), 0.9);

        let mut boxes = vec![tb1, tb2, tb3];
        // Sort top-to-bottom, left-to-right
        boxes.sort_by_key(|b| (b.rect.top(), b.rect.left()));

        assert_eq!(boxes[0].rect.top(), 10);
        assert_eq!(boxes[1].rect.top(), 20);
        assert_eq!(boxes[2].rect.top(), 30);
        // Within same top row, left takes priority
        assert_eq!(boxes[1].rect.left(), 0);
    }
}
