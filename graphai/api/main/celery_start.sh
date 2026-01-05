#!/usr/bin/env bash
set -euo pipefail

echo "=== [GraphAI] Deploying Celery GPU/CPU workers ==="

# ---------------------------------------------------------------------
# 1) Load .env from repo root to get AI_RABBITMQ_DEV_CELERY_USER/PASS
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  echo "Loading environment from $REPO_ROOT/.env"
  # auto-export vars while sourcing
  set -a
  # shellcheck source=/dev/null
  source "$REPO_ROOT/.env"
  set +a
else
  echo "ERROR: .env not found in $REPO_ROOT"
  exit 1
fi

: "${AI_RABBITMQ_DEV_CELERY_USER:?AI_RABBITMQ_DEV_CELERY_USER not set in .env}"
: "${AI_RABBITMQ_DEV_CELERY_PASS:?AI_RABBITMQ_DEV_CELERY_PASS not set in .env}"

export CELERY_BROKER_URL="amqp://${AI_RABBITMQ_DEV_CELERY_USER}:${AI_RABBITMQ_DEV_CELERY_PASS}@localhost:5672//"
export CELERY_RESULT_BACKEND="redis://localhost:6379/1"

echo "Broker:  amqp://${AI_RABBITMQ_DEV_CELERY_USER}:***@localhost:5672//"
echo "Backend: $CELERY_RESULT_BACKEND"

# Make sure we're in api/main so `-A main.celery_instance` resolves correctly
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------
# 1b) Add micromamba tools env to PATH
# ---------------------------------------------------------------------
TOOLS_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
TOOLS_BIN="$TOOLS_ROOT/envs/tools/bin"

if [[ -d "$TOOLS_BIN" ]]; then
  export PATH="$TOOLS_BIN:$PATH"
  echo "Using external tools from: $TOOLS_BIN"
else
  echo "WARN: tools env not found at $TOOLS_BIN; ffmpeg/tesseract/etc may be missing" >&2
fi

# ---------------------------------------------------------------------
# 2) Stop existing Celery workers
# ---------------------------------------------------------------------
echo "=== [GraphAI] Stopping existing Celery workers ==="
PIDS="$(pgrep -f 'celery.*main.celery_instance' || true)"

if [ -n "$PIDS" ]; then
  echo "Killing PIDs: $PIDS"
  kill $PIDS || true
  sleep 3

  PIDS="$(pgrep -f 'celery.*main.celery_instance' || true)"
  if [ -n "$PIDS" ]; then
    echo "Force killing remaining PIDs: $PIDS"
    kill -9 $PIDS || true
  fi
else
  echo "No existing Celery workers found."
fi

# ---------------------------------------------------------------------
# 3) Start new workers (1 worker per GPU + 1 CPU worker)
# ---------------------------------------------------------------------
echo "=== [GraphAI] Starting new Celery workers ==="

#-------------#
# GPU workers #
#-------------#

# embedding_gpu ......... 🔴 Misson critical (for Chatbot)
# voice_gpu ............. 🔵 Graph pipeline work (slow tasks)
# translation_gpu ....... 🔵 Graph pipeline work (slow tasks)

#-------------#
# CPU workers #
#-------------#

# caching ............... 🔴 Highest priority (cache hits)
# rag ................... 🔴 Misson critical (for Chatbot)
# image ................. 🔴 High priority (for RAG pipeline)

# text .................. 🟡 Concept detection (lots of tasks)

# video ................. 🔵 Graph pipeline work
# voice ................. 🔵 Graph pipeline work
# translation ........... 🔵 Graph pipeline work

# ontology .............. 🟢 Used sparingly
# scraping .............. 🟢 Used sparingly
# celery ................ ⚪ Lowest priority fallback queue

#--------------------------#
# Launch GPU workers group #
#--------------------------#

# 🔴 embedding_gpu (GPU #0)
echo "---------------------------------------------"
echo "🚀 Launching "embedding_gpu" worker on GPU #0"
echo "---------------------------------------------"
CUDA_VISIBLE_DEVICES=0 celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -Q embedding_gpu --concurrency=1 -n RCP_GPU0_Embedding &
# echo "✅ Worker launched on GPU #0."
echo

# 🔵 voice_gpu (GPU #1)
echo "-----------------------------------------"
echo "🚀 Launching "voice_gpu" worker on GPU #1"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=1 celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -Q voice_gpu --concurrency=1 -n RCP_GPU1_Voice &
# echo "✅ Worker launched on GPU #1."
echo

# 🔵 voice_gpu (GPU #2)
echo "-----------------------------------------"
echo "🚀 Launching "voice_gpu" worker on GPU #2"
echo "-----------------------------------------"
CUDA_VISIBLE_DEVICES=2 celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -Q voice_gpu --concurrency=1 -n RCP_GPU2_Voice &
# echo "✅ Worker launched on GPU #2."
echo

# 🔵 translation_gpu (GPU #3)
echo "-----------------------------------------------"
echo "🚀 Launching "translation_gpu" worker on GPU #3"
echo "-----------------------------------------------"
CUDA_VISIBLE_DEVICES=3 celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -Q translation_gpu --concurrency=1 -n RCP_GPU3_Translation &
# echo "✅ Worker launched on GPU #3."
echo

#--------------------------#
# Launch CPU workers group #
#--------------------------#
unset CUDA_VISIBLE_DEVICES

# 🔴 caching, rag, image (8 CPUs)
echo "---------------------------------------------------"
echo "🚀 Launching "caching, rag, image" worker on 8 CPUs"
echo "---------------------------------------------------"
celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -P prefork -c 8 -Q caching,rag,image -n RCP_CPUx8_Cache_Rag_Img &
# echo "✅ Worker launched on 8 CPUs."
echo

# 🟡 text (6 CPUs)
echo "------------------------------------"
echo "🚀 Launching "text" worker on 6 CPUs"
echo "------------------------------------"
celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -P prefork -c 6 -Q text -n RCP_CPUx6_Text &
# echo "✅ Worker launched on 6 CPUs."
echo

# 🔵 video, voice, translation (4 CPUs)
echo "---------------------------------------------------------"
echo "🚀 Launching "video, voice, translation" worker on 4 CPUs"
echo "---------------------------------------------------------"
celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -P prefork -c 4 -Q video,voice,translation -n RCP_CPUx6_Video_Voice_Transl &
# echo "✅ Worker launched on 4 CPUs."
echo

# 🟢 ontology, scraping, ⚪ celery (2 CPUs)
echo "----------------------------------------------------------"
echo "🚀 Launching "ontology, scraping, celery" worker on 2 CPUs"
echo "----------------------------------------------------------"
celery --broker="$CELERY_BROKER_URL" -A main.celery_instance worker -l info -P prefork -c 2 -Q ontology,scraping,celery -n RCP_CPUx6_Ontl_Scrp_Celery &
# echo "✅ Worker launched on 2 CPUs."
echo

#--------------------------#

# echo "All workers ready to rock and roll! 😎 🎸"
# echo

wait
