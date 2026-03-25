#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-epflgraph/graphai:cpu}"
MODEL_ROOT="${MODEL_ROOT:-$PWD/models}"
WHISPER_MODEL_TYPE="${WHISPER_MODEL_TYPE:-base}"
FASTTEXT_DIM="${FASTTEXT_DIM:-30}"

DOWNLOAD_HF="${DOWNLOAD_HF:-1}"
DOWNLOAD_WHISPER="${DOWNLOAD_WHISPER:-1}"
DOWNLOAD_FASTTEXT="${DOWNLOAD_FASTTEXT:-1}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH." >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}/huggingface" "${MODEL_ROOT}/whisper" "${MODEL_ROOT}/fasttext"

DOCKER_ENV_ARGS=(
  -e HF_HOME=/models/huggingface
  -e SENTENCE_TRANSFORMERS_HOME=/models/huggingface
  -e TRANSFORMERS_CACHE=/models/huggingface
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
fi

run_in_container() {
  local cmd="$1"
  docker run --rm \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${MODEL_ROOT}:/models" \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "${cmd}"
}

echo "Using image: ${IMAGE}"
echo "Model root: ${MODEL_ROOT}"

if [[ "${DOWNLOAD_HF}" == "1" ]]; then
  echo "[1/3] Downloading Hugging Face models..."
  run_in_container "python /app/docker/preload_models.py --cache-dir /models/huggingface"
else
  echo "[1/3] Skipping Hugging Face download (DOWNLOAD_HF=${DOWNLOAD_HF})"
fi

if [[ "${DOWNLOAD_WHISPER}" == "1" ]]; then
  echo "[2/3] Downloading Whisper model (${WHISPER_MODEL_TYPE})..."
  run_in_container "python -c \"import whisper; whisper.load_model('${WHISPER_MODEL_TYPE}', download_root='/models/whisper')\""
else
  echo "[2/3] Skipping Whisper download (DOWNLOAD_WHISPER=${DOWNLOAD_WHISPER})"
fi

if [[ "${DOWNLOAD_FASTTEXT}" == "1" ]]; then
  echo "[3/3] Downloading and reducing fastText models (dim=${FASTTEXT_DIM})..."
  run_in_container "fasttext-reduce --root_dir /models/fasttext --lang en --dim ${FASTTEXT_DIM} && fasttext-reduce --root_dir /models/fasttext --lang fr --dim ${FASTTEXT_DIM}"
else
  echo "[3/3] Skipping fastText download (DOWNLOAD_FASTTEXT=${DOWNLOAD_FASTTEXT})"
fi

echo
echo "Model directories created:"
du -sh "${MODEL_ROOT}/"* 2>/dev/null || true
echo "Done."
