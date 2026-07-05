# OCR API — Design Specification

基于 [rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) (ocr-rs v2.3.1) 封装的 HTTP API 服务，提供文本检测与识别能力，支持 Docker 单机部署。

## 1. Architecture

```
┌──────────────┐     HTTP/JSON     ┌──────────────────────────────────────────┐
│   Client     │ ◄──────────────►  │        Axum HTTP Server (:8080)          │
│  (curl/App/  │                   │                                          │
│   script)    │                   │  ┌────────────┐  ┌────────────────────┐  │
└──────────────┘                   │  │  ocr/*     │──│  Arc<OcrEngine>     │  │
                                   │  │  health    │  │  (PP-OCRv6 medium) │  │
                                   │  └────────────┘  │  CPU or CUDA GPU   │  │
                                   │                    └────────────────────┘  │
                                   └──────────────────────────────────────────┘
```

- **HTTP 框架**: Axum，基于 Tokio 异步运行时
- **OCR 引擎**: `ocr-rs` v2.3.1 + MNN + PP-OCRv6 medium
- **引擎生命周期**: 服务启动时在 `main()` 中初始化 `OcrEngine`，通过 `Arc<T>` 注入 Axum Router。无需懒加载，模型就绪后才启动 server。
- **并发控制**: 通过 `Tokio::sync::Semaphore` 限制最大同时处理的请求数，避免 CPU 过载。
- **GPU 策略**: 构建时通过 Cargo feature 区分 `cpu` / `cuda`，编译出两个独立二进制。GPU 镜像硬编码 `Backend::CUDA`，在 NVIDIA A30 上运行。

## 2. API 接口

### 2.1 单图 OCR

```
POST /ocr
```

**请求** — multipart/form-data:

```http
Content-Type: multipart/form-data

file: <image binary>
```

**请求** — application/json:

```json
{
  "image": "<base64 encoded image>"
}
```

**响应 200**:

```json
{
  "success": true,
  "data": [
    {
      "text": "识别的文字",
      "confidence": 0.95,
      "bbox": {
        "left": 100,
        "top": 50,
        "width": 200,
        "height": 30
      }
    }
  ]
}
```

`confidence` 为识别置信度（非检测置信度）。`bbox` 为 axis-aligned 整数坐标，不包含旋转角点信息。

### 2.2 批量 OCR

```
POST /ocr/batch
```

**请求** — application/json:

```json
{
  "images": [
    "<base64 image 1>",
    "<base64 image 2>"
  ]
}
```

**响应 200**:

```json
{
  "success": true,
  "data": [
    [
      { "text": "第一张图文字1", "confidence": 0.95, "bbox": { ... } },
      { "text": "第一张图文字2", "confidence": 0.88, "bbox": { ... } }
    ],
    [
      { "text": "第二张图文字1", "confidence": 0.92, "bbox": { ... } }
    ]
  ]
}
```

处理策略：**串行处理**——按顺序逐张识别，避免多张同时推理导致的 GPU/CPU 争抢。单图内部的文本区域识别仍然使用 MNN 多线程 + Rayon 并行。

### 2.3 健康检查

```
GET /health
```

**响应 200**:

```json
{
  "status": "ok",
  "version": "2.3.1",
  "model": "PP-OCRv6_medium",
  "backend": "cpu",
  "uptime_secs": 3600
}
```

GPU 模式下 `backend` 返回 `"cuda"`。

### 2.4 错误响应

所有接口失败时返回统一格式：

```json
{
  "success": false,
  "error": "描述错误原因的消息"
}
```

HTTP 状态码:

| 情况 | 状态码 |
|---|---|
| 请求参数错误（无图片、格式不对） | 400 Bad Request |
| 图片超过 payload 限制 | 413 Payload Too Large |
| 图片无法解码 | 400 Bad Request |
| 模型未就绪 | 503 Service Unavailable |
| OCR 识别内部错误 | 500 Internal Server Error |

## 3. 项目结构

```
ocr-api/
├── Cargo.toml
├── .gitignore
├── src/
│   └── main.rs              # 完整的 API 服务代码
├── models/                   # 本地缓存的模型文件（gitignore，首次运行 download-models.sh 生成）
│   ├── PP-OCRv6_medium_det.mnn
│   ├── PP-OCRv6_medium_rec.mnn
│   └── ppocr_keys_v6_medium.txt
├── download-models.sh        # 从 GitHub 拉取模型文件到本地 models/
├── Dockerfile                # CPU 版本
├── Dockerfile.gpu            # GPU 版本 (CUDA)
└── README.md
```

`main.rs` 包含全部业务逻辑（API 层非常薄，本质是 ocr-rs 的 HTTP 胶水层），不分层：

```
main.rs 结构:
├── main()                    — 加载配置 → 初始化引擎 → 启动 Axum
├── AppState                  — 共享状态 (Arc<OcrEngine>, Semaphore, 启动时间)
├── handlers/
│   ├── ocr_handler()         — POST /ocr
│   ├── ocr_batch_handler()   — POST /ocr/batch
│   └── health_handler()      — GET /health
├── helpers/
│   ├── decode_image()        — 处理 multipart 和 base64 输入
│   └── resize_if_needed()    — 超限缩放
└── error/
    └── AppError              — 统一的错误类型，自动映射到 JSON 响应
```

## 4. Cargo.toml 及依赖

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
uuid = { version = "1", features = ["v4"] }
```

- `cpu` feature：默认，`Backend::CPU`
- `cuda` feature：启用 `ocr-rs/cuda`（MNN 编译时 `MNN_CUDA=ON`），`Backend::CUDA`
- 构建命令：`cargo build --release` (CPU) 或 `cargo build --release --features cuda` (GPU)

## 5. 配置

全部通过环境变量注入：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OCR_HOST` | `0.0.0.0` | 监听地址 |
| `OCR_PORT` | `8080` | 监听端口 |
| `OCR_DET_MODEL` | `models/PP-OCRv6_medium_det.mnn` | 检测模型路径 |
| `OCR_REC_MODEL` | `models/PP-OCRv6_medium_rec.mnn` | 识别模型路径 |
| `OCR_CHARSET` | `models/ppocr_keys_v6_medium.txt` | 字符集文件路径 |
| `OCR_THREADS` | `4` | MNN 推理线程数 |
| `OCR_CONCURRENCY` | `10` | 最大并发请求数 |
| `OCR_CONFIDENCE` | `0.5` | 最低置信度阈值 |
| `OCR_MAX_IMAGE_SIZE` | `4096` | 图片最大边长（超限自动等比缩放） |
| `OCR_MAX_PAYLOAD_SIZE` | `20971520` (20MB) | 单次请求最大 body 字节数 |
| `RUST_LOG` | `info` | 日志级别 |

## 6. 日志

使用 `env_logger`，通过 `RUST_LOG` 控制级别：

- `info` — 每条请求一行：`METHOD /path - STATUS - N regions - T seconds`
- `debug` — 增加每张图的识别细节
- `error` — 仅错误

示例日志行:

```
[2026-07-05 12:00:00] POST /ocr - 200 OK - 3 text regions - 1.23s
[2026-07-05 12:00:00] POST /ocr/batch - 200 OK - 5 images, 12 text regions total - 4.56s
```

## 7. 模型文件

模型文件来自 rust-paddle-ocr 仓库（git 跟踪），直接从 GitHub raw 下载：

| 文件 | 来源 | 大小 |
|---|---|---|
| `PP-OCRv6_medium_det.mnn` | 检测模型 | ~30 MB |
| `PP-OCRv6_medium_rec.mnn` | 识别模型 | ~37 MB |
| `ppocr_keys_v6_medium.txt` | 字符集 | ~75 KB |

**获取方式**：首次运行 `download-models.sh`，通过 `curl` 从 `github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/` 下载到本地 `models/` 目录。`.gitignore` 忽略 `models/`，模型不进入版本控制。

**Docker 构建**：`COPY models/ /app/models/` 将本地缓存的模型文件打包进镜像，不重复下载。

## 8. 后端选择（GPU 策略）

- **CPU 镜像**：默认 `Backend::CPU`，使用 Debian slim 基础镜像
- **GPU 镜像**：硬编码 `Backend::CUDA`，使用 CUDA 12.4 基础镜像，在 NVIDIA A30 上运行
- 策略在编译期通过 feature 决定，不支持运行时切换

代码中通过 `#[cfg]` 选择后端：

```rust
#[cfg(feature = "cuda")]
let backend = Backend::CUDA;
#[cfg(not(feature = "cuda"))]
let backend = Backend::CPU;
```

## 9. Docker 构建

### 9.1 CPU 版本

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

### 9.2 GPU 版本 (CUDA)

```dockerfile
FROM nvidia/cuda:12.4-devel-bookworm AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y cmake pkg-config && rm -rf /var/lib/apt/lists/*
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

### 9.3 使用方式

```bash
# 1. 下载模型（只需执行一次）
./download-models.sh

# 2. CPU 构建运行
docker build -t ocr-api:cpu .
docker run -d -p 8080:8080 ocr-api:cpu

# 3. GPU 构建运行（需要 nvidia-container-toolkit）
docker build -t ocr-api:gpu -f Dockerfile.gpu .
docker run -d -p 8080:8080 --gpus all ocr-api:gpu
```

## 10. 请求保护机制

| 防护层 | 实现方式 |
|---|---|
| Payload 上限 | Axum 的 `ContentLengthLimit` 或手动检查 Content-Length |
| 图片尺寸上限 | 解码后检查宽高，超过 `OCR_MAX_IMAGE_SIZE` 时等比缩放 |
| 并发上限 | `Semaphore::acquire()` 控制同时处理的请求数 |
| 无效图片 | 解码失败返回 400 |

## 11. 测试策略

API 层测试（单元测试和集成测试混合）：

- 辅助函数测试 — `decode_image`, `resize_if_needed`
- 请求处理测试 — 用 `axum::test` 模拟请求验证路由和响应格式
- 健康检查 — 验证 `/health` 返回正确的模型信息
- 错误响应 — 验证无效输入返回正确状态码和格式

OCR 精度测试不在本项目范围（由上游 ocr-rs 库覆盖）。
