# PP-OCRv6 Server

> 为**内网环境**设计的高性能 OCR API 服务，基于 PP-OCRv6 + Rust，一张图不到 1 秒。

[![Docker CPU](https://img.shields.io/docker/pulls/wzzxx/paddle-ocr-v6-server?label=cpu%20image)](https://hub.docker.com/r/wzzxx/paddle-ocr-v6-server)
[![Docker GPU](https://img.shields.io/docker/pulls/wzzxx/paddle-ocr-v6-server?label=gpu%20image)](https://hub.docker.com/r/wzzxx/paddle-ocr-v6-server)
[![GitHub](https://img.shields.io/github/license/wzzxx-wang/paddle-ocr-v6-server)](https://github.com/wzzxx-wang/paddle-ocr-v6-server)

---

## 目录

- [为什么选择这个项目](#为什么选择这个项目)
- [性能](#性能)
- [一键部署](#一键部署)
  - [CPU 部署](#cpu-部署)
  - [GPU 部署（推荐）](#gpu-部署推荐)
- [API 使用](#api-使用)
- [配置参考](#配置参考)
- [离线部署（无互联网环境）](#离线部署无互联网环境)
- [从源码构建](#从源码构建)
- [项目结构](#项目结构)
- [License](#license)

---

## 为什么选择这个项目

- **⚡ GPU 加速，毫秒级响应** — 基于 MNN 推理引擎 + CUDA 加速，RTX 5060 上选 medium 模型，一张图 **不到 1 秒** 完成识别
- **🏭 专为内网设计** — 无外部依赖调用，不需要连接任何互联网服务，纯 HTTP API，离线部署包开箱即用
- **📦 一键部署，开箱即用** — DockerHub 提供预构建镜像，`docker run` 一条命令启动服务
- **🎯 三档模型按需选择** — `medium`（默认，精度优先）、`small`（均衡）、`tiny`（极速），请求时动态切换，无需重启
- **🔀 单图多行并发识别** — 检测到的每个文本行独立并发推理，充分利用 GPU 并行能力
- **🔄 智能 LRU 缓存** — 相同图片重复请求零延迟，基于真实字节大小驱逐，不会 OOM
- **📤 双输入模式** — 支持文件上传（multipart）和 base64（JSON body），适配各种调用方
- **📋 单张 & 批量** — `/ocr` 单张识别，`/ocr/batch` 批量识别，批量内图片级并发
- **⚙️ 运行时动态配参** — 并发数、线程数、置信度阈值等全部通过环境变量配置

## 性能

| 显卡 | 模型 | 单图耗时 | 适用场景 |
|------|------|----------|----------|
| RTX 5060 | medium | **~0.8–1.2s** | 标清文档、密集文字 |
| RTX 5060 | small | **~0.4–0.6s** | 一般图文 |
| RTX 5060 | tiny | **~0.2–0.3s** | 高速流水线 |
| 纯 CPU | medium | ~3–8s（取决于核心数） | 无 GPU 环境兜底 |

> 测试条件：默认参数，`OCR_THREADS=4`，`OCR_INFER_CONCURRENCY=4`，分辨率 1920×1080 左右的中文文档图片。

## 一键部署

镜像已发布到 DockerHub，无需编译，一条命令启动。

### CPU 部署

适合低并发、无 GPU 的内网环境：

```bash
# 拉取镜像
docker pull wzzxx/paddle-ocr-v6-server:latest

# 准备模型文件
mkdir models && cd models
# 从 release 下载或 scp 上传模型文件到 models/ 目录

# 启动服务
docker run -d -p 8080:8080 \
  --name ocr-api \
  -v $(pwd)/models:/app/models \
  wzzxx/paddle-ocr-v6-server:latest
```

或用 docker-compose：

```yaml
# docker-compose.yml
version: "3.9"
services:
  ocr-api:
    image: wzzxx/paddle-ocr-v6-server:latest
    container_name: ocr-api
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    environment:
      OCR_CONCURRENCY: "10"
      OCR_CACHE_MAX_MB: "500"
    restart: unless-stopped
```

```bash
docker compose up -d
```

### GPU 部署（推荐）

NVIDIA 驱动 535+ + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 已安装即可：

```bash
# 拉取 GPU 镜像
docker pull wzzxx/paddle-ocr-v6-server:gpu

# 启动服务
docker run -d -p 8080:8080 \
  --name ocr-api \
  --gpus all \
  -v $(pwd)/models:/app/models \
  wzzxx/paddle-ocr-v6-server:gpu
```

或用 docker-compose：

```yaml
# docker-compose.yml
version: "3.9"
services:
  ocr-api:
    image: wzzxx/paddle-ocr-v6-server:gpu
    container_name: ocr-api
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    environment:
      OCR_CONCURRENCY: "10"
      OCR_CACHE_MAX_MB: "500"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
docker compose up -d
```

### 快速验证

```bash
curl http://localhost:8080/health
```

响应示例：

```json
{
  "status": "ok",
  "version": "2.3.1",
  "model": "PP-OCRv6_medium",
  "backend": "cuda",
  "uptime_secs": 3600
}
```

`"backend": "cuda"` 表示 GPU 模式运行正常。

> **模型文件说明**：镜像内**不内置**模型文件（~85MB），需单独挂载。可从 [GitHub Release](https://github.com/zibo-chen/rust-paddle-ocr/releases/tag/v2.3.1) 下载，或使用 [`download-models.sh`](download-models.sh) 自动下载。

---

## API 使用

### `POST /ocr` — 单张图片识别

**方式一：文件上传（multipart）**

```bash
curl -X POST http://localhost:8080/ocr \
  -F "file=@document.png" \
  -F "model=medium"
```

**方式二：base64 编码（JSON）**

```bash
curl -X POST http://localhost:8080/ocr \
  -H "Content-Type: application/json" \
  -d '{"image": "'$(base64 -w0 document.png)'", "model": "small"}'
```

**方式三：URL 查询参数指定模型**

```bash
curl -X POST "http://localhost:8080/ocr?model=tiny" \
  -F "file=@document.png"
```

**参数说明**

| 参数 | 位置 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | multipart | 文件 | — | 图片文件（multipart 模式必填） |
| `image` | JSON body | string | — | base64 编码的图片（JSON 模式必填） |
| `model` | query / JSON / multipart | string | `"medium"` | 模型变体：`medium`、`small`、`tiny` |

**响应示例**

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

### `POST /ocr/batch` — 批量识别

仅支持 JSON 格式。未命中缓存的图片并发处理（由 `OCR_CONCURRENCY` 控制并发度）。

```bash
curl -X POST http://localhost:8080/ocr/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64_image1>", "<base64_image2>"],
    "model": "medium"
  }'
```

**响应示例**

```json
{
  "success": true,
  "data": [
    [{ "text": "第一张图片的结果", "confidence": 0.95, "bbox": { "left": 10, "top": 20, "width": 200, "height": 30 } }],
    [{ "text": "第二张图片的结果", "confidence": 0.93, "bbox": { "left": 5, "top": 15, "width": 180, "height": 25 } }]
  ]
}
```

### `GET /health` — 健康检查

```bash
curl http://localhost:8080/health
```

## 配置参考

通过环境变量配置，支持 `.env` 文件自动加载。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OCR_HOST` | `0.0.0.0` | 监听地址（内网部署建议保持默认） |
| `OCR_PORT` | `8080` | 监听端口 |
| `OCR_THREADS` | `4` | 推理线程数（CPU 模式有效，GPU 模式下自动调优） |
| `OCR_CONFIDENCE` | `0.5` | 识别置信度阈值（低于此值的结果被过滤） |
| `OCR_CONCURRENCY` | `10` | 最大并发请求数（内网调用方多时可调高） |
| `OCR_INFER_CONCURRENCY` | `4` | 单张图片内文本行识别的最大并发数 |
| `OCR_CACHE_MAX_MB` | `500` | LRU 缓存上限（MB），超出后驱逐最旧条目 |
| `OCR_MAX_IMAGE_SIZE` | `4096` | 图片最大宽/高（像素），超出等比缩放到此限制 |
| `OCR_MAX_PAYLOAD_SIZE` | `20971520` | 请求体最大字节数（20 MiB） |

```bash
# 启动时指定参数
docker run -d -p 8080:8080 \
  --name ocr-api \
  --gpus all \
  -v $(pwd)/models:/app/models \
  -e OCR_CONCURRENCY=20 \
  -e OCR_CACHE_MAX_MB=1000 \
  -e OCR_CONFIDENCE=0.3 \
  wzzxx/paddle-ocr-v6-server:gpu
```

## 离线部署（无互联网环境）

### 方案一：预拉取镜像

```bash
# 在有网络的机器上拉取（只需一次）
docker pull wzzxx/paddle-ocr-v6-server:latest

# 导出为 tar 文件
docker save wzzxx/paddle-ocr-v6-server:latest | gzip > paddle-ocr-cpu.tar.gz

# 复制到内网机器，然后加载
gunzip -c paddle-ocr-cpu.tar.gz | docker load
```

### 方案二：完整离线包

在 [`deploy/`](deploy/) 目录下提供了一键部署脚本和 docker-compose 配置。将以下文件打包传输到内网机器：

```
deploy/
├── docker-compose.cpu.yml    # CPU 部署配置
├── docker-compose.gpu.yml    # GPU 部署配置
├── models/                   # OCR 模型文件目录
│   ├── PP-OCRv6_medium_det.mnn
│   ├── PP-OCRv6_medium_rec.mnn
│   ├── PP-OCRv6_small_det.mnn
│   ├── PP-OCRv6_small_rec.mnn
│   ├── PP-OCRv6_tiny_det.mnn
│   ├── PP-OCRv6_tiny_rec.mnn
│   ├── ppocr_keys_v6_medium.txt
│   ├── ppocr_keys_v6_small.txt
│   └── ppocr_keys_v6_tiny.txt
```

将镜像 tar 包和 `deploy/` 目录复制到内网机器后：

```bash
# 1. 加载镜像
gunzip -c paddle-ocr-cpu.tar.gz | docker load

# 2. 启动服务（模型和配置文件在 deploy/ 中已就绪）
cd deploy
docker compose -f docker-compose.cpu.yml up -d

# 3. 验证
curl http://localhost:8080/health
```

---

## 从源码构建

> 适用于需要二次开发、自定义镜像，或网络限制下使用 vendored MNN 的场景。

### 前置条件

- Rust 1.75+
- CMake + pkg-config + build-essential + libclang-dev
- Git（build.rs 会克隆 MNN 源码）

### 构建并运行

```bash
# 克隆仓库
git clone https://github.com/wzzxx-wang/paddle-ocr-v6-server.git
cd paddle-ocr-v6-server/ocr-api

# 下载模型文件
./download-models.sh

# CPU 构建（默认）
cargo build --release
./target/release/ocr-api

# GPU 构建（需要 CUDA 12.4+）
cargo build --release --features cuda

# 运行测试
cargo test
```

### Docker 镜像构建

```bash
# CPU 镜像
docker build -t paddle-ocr-v6-server:cpu .

# GPU 镜像（需要 CUDA builder 镜像）
docker build -t paddle-ocr-v6-server:gpu -f Dockerfile.gpu .
```

### 网络受限环境构建

`build.rs` 会从 GitHub 克隆 MNN（~80MB）。如果内网编译遇到网络问题，有两种解决方式：

1. **使用 vendored MNN** — 在项目中预置 MNN 源码，参考 [CI 工作流](.github/workflows/docker.yml) 的 build-bundle 策略
2. **配置代理** — 为 Git 和 Cargo 设置内网可用的 mirror

---

## 项目结构

```
paddle-ocr-v6-server/
├── ocr-api/                    # 核心服务
│   ├── src/main.rs             # 全部代码（~850 行，单体文件）
│   ├── Cargo.toml              # 依赖声明
│   ├── Dockerfile              # CPU 镜像构建
│   ├── Dockerfile.gpu          # GPU 镜像构建
│   ├── download-models.sh      # 模型下载脚本
│   └── deploy/                 # 离线部署包
├── .github/workflows/          # CI/CD
│   └── docker.yml              # DockerHub 自动构建
├── LICENSE                     # MIT
└── README.md
```

### 架构流程

```
请求进入 → 获取并发信号量 → 解析请求（multipart/JSON）
  → 检查 LRU 缓存 → 未命中 → 图片解码
  → OCR 检测 → 多行并发识别 → 排序过滤
  → 存入缓存 → 返回 JSON 响应
```

- **信号量控制** — `OCR_CONCURRENCY` 限制全局并发请求数，防止高并发下 OOM
- **文本行并发** — 单张图片内的多个文本行通过 `OCR_INFER_CONCURRENCY` 控制并行度
- **LRU 缓存** — 以 `hash(原始图片字节 + 模型名)` 为 key，基于真实字节大小驱逐，避免缓存 OOM
- **优雅关闭** — 收到 SIGINT/SIGTERM 后等待进行中的 OCR 完成再退出

## License

MIT
