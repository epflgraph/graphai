#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-epflgraph}"
REPOSITORY="${REPOSITORY:-graphai}"
IMAGE_NAME="${NAMESPACE}/${REPOSITORY}"
TAG="${1:-latest}"
PRELOAD_MODELS="${PRELOAD_MODELS:-1}"
TARGET="${TARGET:-base-cpu}"

echo "Building image: ${IMAGE_NAME}:${TAG} (target=${TARGET})"
docker build \
  --target "${TARGET}" \
  --build-arg "PRELOAD_MODELS=${PRELOAD_MODELS}" \
  -t "${IMAGE_NAME}:${TAG}" \
  .

echo "Running smoke test container..."
docker run --rm --entrypoint python "${IMAGE_NAME}:${TAG}" --version >/dev/null

if ! docker info 2>/dev/null | grep -q "Username:"; then
  echo "Docker Hub login required. Opening docker login..."
  docker login
fi

if [ "${TAG}" != "latest" ]; then
  echo "Tagging ${IMAGE_NAME}:${TAG} as ${IMAGE_NAME}:latest"
  docker tag "${IMAGE_NAME}:${TAG}" "${IMAGE_NAME}:latest"
fi

echo "Pushing ${IMAGE_NAME}:${TAG}"
docker push "${IMAGE_NAME}:${TAG}"

if [ "${TAG}" != "latest" ]; then
  echo "Pushing ${IMAGE_NAME}:latest"
  docker push "${IMAGE_NAME}:latest"
fi

echo "Publish complete."
echo "Docker Hub URL: https://hub.docker.com/r/${IMAGE_NAME}"
