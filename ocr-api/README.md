# OCR API

基于 [rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) (PP-OCRv6) 的高性能 OCR HTTP API 服务，使用 Rust + Axum 构建。

## 特性

- **多模型运行时切换** — 内置 medium / small / tiny 三种 PP-OCRv6 模型，请求时按需选择
- **并发文本行识别** — 检测到的每个文本行独立并发识别，通过 `OCR_INFER_CONCURRENCY` 控制并行度
- **大小感知的 LRU 缓存** — 以图片字节 + 模型名为 key，基于真实字节大小驱逐，避免计数缓存的内存溢出
- **双输入模式** — 支持 `multipart/form-data`（文件上传）和 `application/json`（base64）两种请求
- **单张 / 批量 OCR** — `/ocr` 单张识别，`/ocr/batch` 批量识别，批量请求内图片级并发
- **优雅关闭** — 支持 SIGINT / SIGTERM 信号，等待进行中的 OCR 完成后退出
- **云原生** — 提供 CPU 和 GPU（CUDA）两种 Docker 镜像

## 架构

```
                  ┌──────────────────────────────┐
                  │         Axum Router           │
                  │  POST /ocr  POST /ocr/batch   │
                  │         GET /health           │
                  └──────────┬───────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │        AppState (Arc)        │
              │  ┌──────┐  ┌──────┐ ┌──────┐│
              │  │Model │  │Concur│ │Cache ││
              │  │ Set  │  │Semap │ │ OCr  ││
              │  └──────┘  └──────┘ └──────┘│
              └──────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              │     ModelSet        │
              │ medium / small/tiny │
              │ (Arc<OcrEngine> x3)│
              └────────────────────┘
```

### 请求处理流程

1. 请求进入，获取并发信号量（`OCR_CONCURRENCY`）许可
2. 解析请求体（multipart 或 JSON base64），提取原始图片字节和模型名
3. 计算缓存 key = `hash(raw_image_bytes + model_name)`，检查缓存
4. 缓存命中 → 直接返回；缓存未命中 → 进入 OCR 管线
5. OCR 管线：
   - `image::load_from_memory` 解码图片
   - 若图片尺寸超过 `OCR_MAX_IMAGE_SIZE`，按比例缩放到限制内
   - `OcrEngine::detect()` 检测文本区域（`spawn_blocking`）
   - 每个文本区域裁剪后通过信号量（`OCR_INFER_CONCURRENCY`）控制并发的 `recognize_text()` 识别
   - 按置信度阈值过滤，按阅读顺序（从上到下、从左到右）排序
6. 结果存入缓存，返回 JSON 响应

## 快速开始

### 前置条件

- Rust 1.75+（仅编译时需要）
- CMake + pkg-config（编译 `ocr-rs` MNN 引擎时需要）
- 模型文件（运行需要）

### 1. 下载模型

```bash
./download-models.sh
```

脚本会自动从 rust-paddle-ocr v2.3.1 的 GitHub release 下载 medium 模型（det + rec + charset）到 `models/` 目录。

如果需要 small 和 tiny 模型，手动下载并放入 `models/` 目录：

```bash
# small
curl -# -L -o models/PP-OCRv6_small_det.mnn \
  https://github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/PP-OCRv6_small_det.mnn
curl -# -L -o models/PP-OCRv6_small_rec.mnn \
  https://github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/PP-OCRv6_small_rec.mnn
curl -# -L -o models/ppocr_keys_v6_small.txt \
  https://github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/ppocr_keys_v6_small.txt

# tiny
curl -# -L -o models/PP-OCRv6_tiny_det.mnn \
  https://github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/PP-OCRv6_tiny_det.mnn
curl -# -L -o models/PP-OCRv6_tiny_rec.mnn \
  https://github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/PP-OCRv6_tiny_rec.mnn
curl -# -L -o models/ppocr_keys_v6_tiny.txt \
  https://github.com/zibo-chen/rust-paddle-ocr/raw/v2.3.1/models/ppocr_keys_v6_tiny.txt
```

### 2. 配置环境变量

复制 `.env` 文件并调整：

```bash
cp .env .env.local
# 编辑 .env.local 修改配置
```

所有配置项见 [配置](#配置) 章节。

### 3. 编译并运行

```bash
# CPU 模式（默认）
cargo build --release
./target/release/ocr-api

# 开发模式（热启动更快）
cargo run
```

服务默认监听 `0.0.0.0:8080`。

## API 文档

### `GET /health`

健康检查端点。

```bash
curl http://localhost:8080/health
```

**响应示例：**

```json
{
  "status": "ok",
  "version": "2.3.1",
  "model": "PP-OCRv6_medium",
  "backend": "cpu",
  "uptime_secs": 3600
}
```

### `POST /ocr`

对单张图片执行 OCR 识别。

**方式一：multipart/form-data（文件上传）**

```bash
curl -X POST http://localhost:8080/ocr \
  -F "file=@test.png" \
  -F "model=medium"
```

**方式二：application/json（base64 编码）**

```bash
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w0 test.png)'", "model": "small"}'
```

**方式三：URL 查询参数指定模型**

```bash
curl -X POST http://localhost:8080/ocr?model=tiny \
  -F "file=@test.png"
```

**参数说明：**

| 参数 | 位置 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | multipart | 文件 | — | 图片文件（multipart 模式必填） |
| `image` | JSON body | string | — | base64 编码的图片（JSON 模式必填） |
| `model` | query / JSON / multipart | string | `"medium"` | 模型变体：`medium`、`small`、`tiny` |

**响应示例：**

```json
{
  "success": true,
  "data": [
    {
      "text": "你好世界",
      "confidence": 0.97,
      "bbox": {
        "left": 10,
        "top": 20,
        "width": 200,
        "height": 30
      }
    }
  ]
}
```

### `POST /ocr/batch`

批量对多张图片执行 OCR 识别。仅支持 JSON 格式。未命中缓存的图片会并发处理。

```bash
curl -X POST http://localhost:8080/ocr/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64_image1>", "<base64_image2>"],
    "model": "medium"
  }'
```

**响应示例：**

```json
{
  "success": true,
  "data": [
    [{ "text": "第一张图片的结果", "confidence": 0.95, "bbox": { ... } }],
    [{ "text": "第二张图片的结果", "confidence": 0.93, "bbox": { ... } }]
  ]
}
```

## 配置

通过环境变量配置，支持 `.env` 文件自动加载。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_HOST` | `0.0.0.0` | 监听地址 |
| `OCR_PORT` | `8080` | 监听端口 |
| `OCR_THREADS` | `4` | OCR 引擎推理线程数 |
| `OCR_CONFIDENCE` | `0.5` | 识别结果置信度阈值（低于此值过滤） |
| `OCR_CONCURRENCY` | `10` | 最大并发请求数 |
| `OCR_INFER_CONCURRENCY` | `4` | 单张图片内文本行识别的最大并发数 |
| `OCR_CACHE_MAX_MB` | `500` | LRU 缓存最大字节数（MB），超出后驱逐最旧条目 |
| `OCR_MAX_IMAGE_SIZE` | `4096` | 图片最大宽/高（像素），超出等比缩放到此限制 |
| `OCR_MAX_PAYLOAD_SIZE` | `20971520` | 请求体最大字节数（20 MiB） |

## Docker 部署

### CPU 镜像

```bash
# 构建
docker build -t ocr-api:cpu .

# 运行（指定环境变量）
docker run -d -p 8080:8080 \
  -v /path/to/models:/app/models \
  -e OCR_CONCURRENCY=10 \
  -e OCR_CACHE_MAX_MB=500 \
  ocr-api:cpu
```

### GPU 镜像（CUDA）

```bash
# 构建
docker build -t ocr-api:gpu -f Dockerfile.gpu .

# 运行
docker run -d -p 8080:8080 \
  --gpus all \
  -v /path/to/models:/app/models \
  ocr-api:gpu
```

GPU 镜像要求主机已安装 NVIDIA 驱动和 NVIDIA Container Toolkit。

## 缓存机制

服务内置了按字节大小追踪的 LRU 缓存，以 `hash(原始图片字节 + 模型名称)` 为 key：

- **缓存键**：基于原始图片字节（而非解码后图片），确保相同图片不同格式也可命中
- **驱逐策略**：当缓存总大小超过 `OCR_CACHE_MAX_MB` 时，从最旧的条目开始逐出，直到容量满足
- **大小计算**：`size_of::<OcrItem>() * 条目数 + 所有文本字符串的 capacity` 之和
- **线程安全**：通过 `Mutex<OcrCacheInner>` 保护，使用 `lru` crate 的 `LruCache` 配合手动字节追踪
- **边界一致性**：当同一个 key 重复写入时，自动减去旧数据的大小后再执行驱逐逻辑，避免 `current_bytes` 漂移

缓存状态可通过日志观察：响应中包含 `(cached)` 标记的结果来自缓存。

## 开发

### 项目结构

```
ocr-api/
├── Cargo.toml          # 依赖和特性声明
├── Cargo.lock          # 依赖锁定文件
├── .env                # 默认环境变量配置
├── Dockerfile          # CPU Docker 镜像
├── Dockerfile.gpu      # GPU (CUDA) Docker 镜像
├── download-models.sh  # 模型下载脚本
├── src/
│   └── main.rs         # 所有代码（约 850 行，单体文件）
├── models/             # OCR 模型文件目录 (.gitignore 排除)
└── target/             # 编译产出目录
```

### 代码组织

整个服务为单体文件 `src/main.rs`，按功能模块顺序组织：

| 模块 | 行号 | 说明 |
|------|------|------|
| `AppConfig` | 25–61 | 环境变量配置加载 |
| `ModelVariant` / `ModelSet` | 64–140 | 模型变体枚举、路径映射、引擎预加载 |
| `AppState` | 143–152 | 共享状态结构体 |
| `AppError` | 155–189 | 统一错误类型及 HTTP 响应映射 |
| DTOs | 193–255 | 请求/响应数据结构 |
| 图片处理 | 260–274 | 解码、等比缩放 |
| `OcrCache` | 278–340 | 大小感知 LRU 缓存 |
| Multipart 解析 | 343–382 | multipart body 解析 |
| `recognize_text_lines` | 386–453 | OCR 管线：检测 → 并发识别 → 排序 |
| base64 处理 | 456–474 | JSON base64 解码 |
| Handler 函数 | 478–678 | HTTP 请求处理器 |
| `main` | 681–752 | 启动入口、优雅关闭 |
| 单元测试 | 755–847 | 各模块单元测试 |

### 构建特性

```bash
# 默认 CPU 推理
cargo build --release

# CUDA GPU 推理
cargo build --release --features cuda

# 运行测试
cargo test
```

### 测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_resize_if_needed_exceeds_limit

# 查看测试日志输出
cargo test -- --nocapture
```

现有测试覆盖了：图片缩放、OcrItem 转换、base64 编解码（含 data URI 格式）、错误响应、文本框排序等场景。

## 依赖

| Crate | 用途 |
|-------|------|
| `ocr-rs` | PP-OCRv6 推理引擎（MNN 后端） |
| `axum` 0.7 | HTTP 路由和请求处理 |
| `tokio` 1 | 异步运行时 |
| `serde` / `serde_json` | 序列化/反序列化 |
| `image` 0.25 | 图片解码和缩放 |
| `lru` 0.12 | LRU 缓存 |
| `multer` / `bytes` / `tokio-stream` | multipart 解析 |
| `base64` 0.22 | base64 编解码 |
| `dotenvy` 0.15 | `.env` 文件自动加载 |
| `thiserror` | 错误类型派生 |
| `futures` | 异步工具 |

## License

MIT
