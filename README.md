# PP-OCRv6 Server

> 为 **内网环境** 设计的高性能 OCR API 服务，基于 PP-OCRv6 + Rust + MNN。

[![GitHub](https://img.shields.io/github/license/wzzxx-wang/paddle-ocr-v6-server)](https://github.com/wzzxx-wang/paddle-ocr-v6-server)
[![Docker](https://img.shields.io/badge/docker-pull-blue?logo=docker)](https://hub.docker.com/r/wzzxx/paddle-ocr-v6-server)

GPU 加速 — NVIDIA RTX 5060 选 medium 模型，**一张图不到 1 秒**完成识别。

提供预构建 Docker 镜像，一条命令启动服务，纯 HTTP API 接口，专为内网离线环境设计。

---

## 亮点

- **⚡ GPU 加速** — MNN + CUDA，5060 上 1 秒识别一张标清文档图
- **📦 一键部署** — `docker run` 即启，无需编译、无需连接外网
- **🎯 三档模型** — `medium`（精度优先）/ `small`（均衡）/ `tiny`（极速），请求时动态切换
- **🔀 文本行并发** — 检测到的多个文本行独立并行推理，充分利用 GPU
- **🔄 LRU 缓存** — 相同图片重复请求零延迟，按真实字节大小驱逐
- **📤 双输入** — 文件上传（multipart）或 base64（JSON）均支持
- **📋 批量识别** — 单次请求提交多张图，图片级并发处理

## 一键部署

```bash
# CPU
docker run -d -p 8080:8080 -v $(pwd)/models:/app/models \
  wzzxx/paddle-ocr-v6-server:latest

# GPU（推荐）
docker run -d -p 8080:8080 --gpus all -v $(pwd)/models:/app/models \
  wzzxx/paddle-ocr-v6-server:gpu

# 验证
curl http://localhost:8080/health
```

## 快速链接

- [完整文档 →](ocr-api/README.md)
- [API 使用](ocr-api/README.md#post-ocr--单张图片识别)
- [配置参考](ocr-api/README.md#配置参考)
- [离线部署指南](ocr-api/README.md#离线部署无互联网环境)
- [GitHub 仓库](https://github.com/wzzxx-wang/paddle-ocr-v6-server)

## License

MIT
