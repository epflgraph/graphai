#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-epflgraph/graphai:cpu}"
MODEL_ROOT="${MODEL_ROOT:-$PWD/models}"
PLATFORM="${PLATFORM:-}"
WHISPER_MODEL_TYPE="${WHISPER_MODEL_TYPE:-base}"
FASTTEXT_DIM="${FASTTEXT_DIM:-30}"

DOWNLOAD_HF="${DOWNLOAD_HF:-1}"
DOWNLOAD_WHISPER="${DOWNLOAD_WHISPER:-1}"
DOWNLOAD_FASTTEXT="${DOWNLOAD_FASTTEXT:-1}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH." >&2
  exit 1
fi

# Pretty output
if [[ -t 1 ]]; then
  C_RESET="$(tput sgr0)"
  C_BOLD="$(tput bold)"
  C_RED="$(tput setaf 1)"
  C_GREEN="$(tput setaf 2)"
  C_YELLOW="$(tput setaf 3)"
  C_BLUE="$(tput setaf 4)"
else
  C_RESET=""
  C_BOLD=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
fi

log_info() {
  echo "${C_BLUE}ℹ️  $*${C_RESET}"
}

log_ok() {
  echo "${C_GREEN}✅ $*${C_RESET}"
}

log_warn() {
  echo "${C_YELLOW}⚠️  $*${C_RESET}"
}

log_err() {
  echo "${C_RED}❌ $*${C_RESET}" >&2
}

mkdir -p "${MODEL_ROOT}/huggingface" "${MODEL_ROOT}/whisper" "${MODEL_ROOT}/fasttext"

DOCKER_ENV_ARGS=(
  -e HF_HOME=/models/huggingface
  -e SENTENCE_TRANSFORMERS_HOME=/models/huggingface
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ENV_ARGS+=(-e HF_TOKEN="${HF_TOKEN}")
fi

run_in_container() {
  local cmd="$1"
  local platform_args=()
  if [[ -n "${PLATFORM}" ]]; then
    platform_args=(--platform "${PLATFORM}")
  fi
  docker run --rm \
    "${platform_args[@]}" \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${MODEL_ROOT}:/models" \
    --entrypoint bash \
    "${IMAGE}" \
    -lc "${cmd}"
}

HF_MARKER="${MODEL_ROOT}/huggingface/.graphai_hf_preload_done"
WHISPER_FILE="${MODEL_ROOT}/whisper/${WHISPER_MODEL_TYPE}.pt"
FASTTEXT_EN_FILE="${MODEL_ROOT}/fasttext/cc.en.${FASTTEXT_DIM}.bin"
FASTTEXT_FR_FILE="${MODEL_ROOT}/fasttext/cc.fr.${FASTTEXT_DIM}.bin"

echo "${C_BOLD}🚀 GraphAI model setup${C_RESET}"
log_info "Using image: ${IMAGE}"
if [[ -n "${PLATFORM}" ]]; then
  log_info "Platform override: ${PLATFORM}"
fi
log_info "Model root: ${MODEL_ROOT}"
log_info "Force redownload: ${FORCE_DOWNLOAD}"

if [[ "${DOWNLOAD_HF}" == "1" ]]; then
  if [[ "${FORCE_DOWNLOAD}" != "1" && -f "${HF_MARKER}" ]]; then
    log_ok "[1/3] Hugging Face models already prepared (marker found), skipping."
  else
    log_info "[1/3] Downloading Hugging Face models..."
    run_in_container "python /app/docker/preload_models.py --cache-dir /models/huggingface"
    touch "${HF_MARKER}"
    log_ok "[1/3] Hugging Face models ready."
  fi
else
  log_warn "[1/3] Skipping Hugging Face download (DOWNLOAD_HF=${DOWNLOAD_HF})"
fi

if [[ "${DOWNLOAD_WHISPER}" == "1" ]]; then
  if [[ "${FORCE_DOWNLOAD}" != "1" && -f "${WHISPER_FILE}" ]]; then
    log_ok "[2/3] Whisper model '${WHISPER_MODEL_TYPE}' already present, skipping."
  else
    log_info "[2/3] Downloading Whisper model (${WHISPER_MODEL_TYPE})..."
    run_in_container "python -c \"import whisper; whisper.load_model('${WHISPER_MODEL_TYPE}', download_root='/models/whisper')\""
    log_ok "[2/3] Whisper model ready."
  fi
else
  log_warn "[2/3] Skipping Whisper download (DOWNLOAD_WHISPER=${DOWNLOAD_WHISPER})"
fi

if [[ "${DOWNLOAD_FASTTEXT}" == "1" ]]; then
  if [[ "${FORCE_DOWNLOAD}" != "1" && -f "${FASTTEXT_EN_FILE}" && -f "${FASTTEXT_FR_FILE}" ]]; then
    log_ok "[3/3] fastText reduced models already present (dim=${FASTTEXT_DIM}), skipping."
  else
    log_info "[3/3] Downloading and reducing fastText models (dim=${FASTTEXT_DIM})..."
    run_in_container "fasttext-reduce --root_dir /models/fasttext --lang en --dim ${FASTTEXT_DIM} && fasttext-reduce --root_dir /models/fasttext --lang fr --dim ${FASTTEXT_DIM}"
    log_ok "[3/3] fastText models ready."
  fi
else
  log_warn "[3/3] Skipping fastText download (DOWNLOAD_FASTTEXT=${DOWNLOAD_FASTTEXT})"
fi

echo
log_info "Model directories:"
du -sh "${MODEL_ROOT}/"* 2>/dev/null || true
log_ok "Done."
