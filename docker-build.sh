#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-epflgraph/graphai}"
IMAGE_TAG="${1:-latest}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"
TARGET="${TARGET:-base-cpu}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} (target=${TARGET}) ..."
docker build \
  --target "${TARGET}" \
  --build-arg "PRELOAD_MODELS=${PRELOAD_MODELS}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .
echo "Done: ${IMAGE_NAME}:${IMAGE_TAG}"
