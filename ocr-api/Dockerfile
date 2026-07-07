FROM rust:1-slim-bookworm AS builder
WORKDIR /app

# Install build dependencies (build.rs will clone MNN from GitHub automatically)
RUN apt-get update && apt-get install -y cmake pkg-config git build-essential libclang-dev ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy manifests for dependency caching
COPY Cargo.toml Cargo.lock ./

COPY src/ ./src/

RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libgomp1 libatomic1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/ocr-api /usr/local/bin/ocr-api
COPY models/ /app/models/
EXPOSE 8080
CMD ["ocr-api"]
