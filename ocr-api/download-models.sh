#!/bin/bash
set -e

REPO="https://github.com/zibo-chen/rust-paddle-ocr"
TAG="v2.3.1"
MODELS_DIR="models"

mkdir -p "$MODELS_DIR"

echo "Downloading PP-OCRv6 medium detection model..."
curl -# -L -o "$MODELS_DIR/PP-OCRv6_medium_det.mnn" \
  "$REPO/raw/$TAG/models/PP-OCRv6_medium_det.mnn"

echo "Downloading PP-OCRv6 medium recognition model..."
curl -# -L -o "$MODELS_DIR/PP-OCRv6_medium_rec.mnn" \
  "$REPO/raw/$TAG/models/PP-OCRv6_medium_rec.mnn"

echo "Downloading PP-OCRv6 medium charset..."
curl -# -L -o "$MODELS_DIR/ppocr_keys_v6_medium.txt" \
  "$REPO/raw/$TAG/models/ppocr_keys_v6_medium.txt"

echo "Done. Files saved to $MODELS_DIR/"
ls -lh "$MODELS_DIR/"
