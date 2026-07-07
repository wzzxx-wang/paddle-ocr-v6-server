# PP-OCRv6 Server

基于 [rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) (PP-OCRv6) 的高性能 OCR HTTP API 服务，使用 Rust + Axum 构建。

核心代码在 [`ocr-api/`](ocr-api/) 目录下。

## 特性

- **多模型运行时切换** — medium / small / tiny 三种 PP-OCRv6 模型
- **并发文本行识别** — 单图内检测到的文本行并发推理
- **大小感知的 LRU 缓存** — 基于字节大小驱逐，避免 OOM
- **双输入模式** — multipart 文件上传 和 JSON base64
- **单张 / 批量 OCR**
- **CPU / GPU（CUDA）** 双 Docker 镜像

## 快速开始

```bash
cd ocr-api
# 下载模型
./download-models.sh

# 编译运行
cargo run --release
```

更多详情见 [`ocr-api/README.md`](ocr-api/README.md)。

## Docker

```bash
# CPU
docker build -t paddle-ocr-v6-server:cpu ocr-api/

# GPU (CUDA)
docker build -t paddle-ocr-v6-server:gpu -f ocr-api/Dockerfile.gpu ocr-api/
```

## License

MIT
