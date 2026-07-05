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
