FROM rust:1-slim-bookworm AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y cmake pkg-config git build-essential libclang-dev ca-certificates && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY src/ ./src/
COPY config.toml /usr/local/cargo/config.toml

# Fetch deps so git checkouts exist
RUN GIT_SSL_NO_VERIFY=1 CARGO_HTTP_SSL_VERIFY=false cargo fetch

# Pre-place MNN in ocr-rs's 3rd_party so build script skips the git clone
COPY ocr-rs-MNN/ /usr/local/cargo/git/checkouts/rust-paddle-ocr-a5cec83695220861/7368dfa/3rd_party/MNN/

RUN GIT_SSL_NO_VERIFY=1 CARGO_HTTP_SSL_VERIFY=false cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libgomp1 libatomic1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/ocr-api /usr/local/bin/ocr-api
COPY models/ /app/models/
EXPOSE 8080
CMD ["ocr-api"]
