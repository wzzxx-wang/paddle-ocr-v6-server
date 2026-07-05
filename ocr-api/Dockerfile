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
