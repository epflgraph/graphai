# syntax=docker/dockerfile:1.7

# Use an official Python runtime as a parent image
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_ROOT_USER_ACTION=ignore \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    NLTK_DATA=/opt/nltk_data \
    GRAPH_CACHE_ROOT=/var/graphai/storage \
    GRAPH_LOG_ROOT=/var/graphai/logs \
    HF_HOME=/opt/models/huggingface \
    HF_LOCAL_MODEL_ROOT=/opt/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/opt/models/huggingface \
    WHISPER_MODEL_ROOT=/opt/models/whisper \
    FASTTEXT_MODEL_ROOT=/opt/models/fasttext \
    SPACY_MODEL_ROOT=/opt/models/spacy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PATH=/opt/micromamba/envs/tools/bin:$PATH \
    PYTHONPATH=/opt/models/spacy:$PYTHONPATH

# Use bash as the default shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Create app directory
WORKDIR /app

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      bzip2 \
      ca-certificates \
      curl \
      git \
      libchromaprint-tools \
      tini \
      tesseract-ocr-eng \
      tesseract-ocr-fra \
      tesseract-ocr-script-latn \
      libgl1 \
      libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy environment.yml for micromamba installation
COPY environment.yml ./environment.yml

# Install micromamba and create the tools environment
ARG MICROMAMBA_VERSION=2.5.0-2
RUN curl -Ls https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64 \
    -o micromamba && \
    echo "c04571cfb0750e5432d530a3068b8fcd232ebed3133358e056e59a90b9852b00  micromamba" | sha256sum -c - && \
    install -m 0755 micromamba /usr/local/bin/micromamba && \
    rm micromamba && \
    micromamba create -y -n tools -f /app/environment.yml && \
    micromamba clean -a -y

# Copy only packaging metadata and requirement files first.
# This keeps dependency layers cached when application source changes.
COPY constraints.txt pyproject.toml setup.py ./
COPY docker/requirements/ ./docker/requirements/

# Build tooling layer: changes rarely, shared by all following pip installs.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade -c constraints.txt -r docker/requirements/requirements-00-build.txt

# Numeric / ABI layer. Keep before PyTorch and ML so NumPy/Numba changes are explicit.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt -r docker/requirements/requirements-01-numeric.txt

# NVIDIA CUDA runtime layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt \
      -r docker/requirements/requirements-02b-nvidia-cu118.txt \
      --extra-index-url https://download.pytorch.org/whl/cu118

# Giant CUDA/PyTorch layer. Keep isolated for Docker cache efficiency.
# Tesla V100 is sm_70; use CUDA 11.8 wheels that include sm_70 support.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt \
      -r docker/requirements/requirements-02a-torch-cu118.txt \
      --extra-index-url https://download.pytorch.org/whl/cu118

# ML/NLP layer: Transformers, Whisper, sentence-transformers, spaCy, Presidio, etc.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt -r docker/requirements/requirements-03-ml-nlp.txt

# Media/OCR/PDF layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt -r docker/requirements/requirements-04-media-ocr.txt

# Cloud/search/auth/database layer. This contains the Git dependency, so isolate it.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt -r docker/requirements/requirements-05-cloud-search-auth.txt

# API/Celery/runtime layer: smaller and most likely to change.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -c constraints.txt -r docker/requirements/requirements-06-web-celery-api.txt

# Second stage for models
FROM base AS models

# Create necessary directories and set permissions
RUN mkdir -p \
      "${NLTK_DATA}" \
      "${GRAPH_CACHE_ROOT}" \
      "${GRAPH_LOG_ROOT}" \
      "${HF_HOME}" \
      "${WHISPER_MODEL_ROOT}" \
      "${FASTTEXT_MODEL_ROOT}" \
      "${SPACY_MODEL_ROOT}" && \
    chmod -R 755 /opt/models /var/graphai

# Download the necessary models for spaCy and NLTK, ensuring that the required language models and data are available for use in the application
# Models are copied from docker/models; this build does not download spaCy or NLTK data.
COPY docker/models/spacy/ ${SPACY_MODEL_ROOT}/
COPY docker/models/nltk/ ${NLTK_DATA}/

# Download and reduce the FastText models for English and French, then remove the original 300-dimension files in the same layer
# Reduced FastText and Whisper models are copied from docker/models; this build does not download or reduce them.
COPY docker/models/fasttext/ ${FASTTEXT_MODEL_ROOT}/
COPY docker/models/whisper/ ${WHISPER_MODEL_ROOT}/

# Hugging Face models are copied from docker/models; this build does not run preload downloads.
COPY docker/models/huggingface/sentence-transformers/all-MiniLM-L12-v2/ \
     ${HF_HOME}/sentence-transformers/all-MiniLM-L12-v2/

# Hugging Face translation models for English-French and English-Italian pairs, both directions, are copied from docker/models; this build does not run preload downloads.
COPY docker/models/huggingface/Helsinki-NLP/opus-mt-tc-big-en-fr/ \
     ${HF_HOME}/Helsinki-NLP/opus-mt-tc-big-en-fr/

COPY docker/models/huggingface/Helsinki-NLP/opus-mt-tc-big-fr-en/ \
     ${HF_HOME}/Helsinki-NLP/opus-mt-tc-big-fr-en/

COPY docker/models/huggingface/Helsinki-NLP/opus-mt-de-en/ \
     ${HF_HOME}/Helsinki-NLP/opus-mt-de-en/

COPY docker/models/huggingface/Helsinki-NLP/opus-mt-it-en/ \
     ${HF_HOME}/Helsinki-NLP/opus-mt-it-en/

COPY docker/models/huggingface/Helsinki-NLP/opus-mt-en-de/ \
     ${HF_HOME}/Helsinki-NLP/opus-mt-en-de/

COPY docker/models/huggingface/Helsinki-NLP/opus-mt-en-it/ \
     ${HF_HOME}/Helsinki-NLP/opus-mt-en-it/

COPY docker/models/huggingface/Davlan/distilbert-base-multilingual-cased-ner-hrl/ \
     ${HF_HOME}/Davlan/distilbert-base-multilingual-cased-ner-hrl/

COPY docker/models/huggingface/microsoft/mdeberta-v3-base/ \
     ${HF_HOME}/microsoft/mdeberta-v3-base/

COPY docker/models/huggingface/urchade/gliner_multi_pii-v1/ \
     ${HF_HOME}/urchade/gliner_multi_pii-v1/

# Verify that the models have been copied correctly and are accessible in the expected directories, and also check that the spaCy models can be imported without issues.
RUN set -eux; \
    test -d "${SPACY_MODEL_ROOT}/en_core_web_sm"; \
    test -d "${SPACY_MODEL_ROOT}/fr_core_news_sm"; \
    test -d "${NLTK_DATA}/corpora/stopwords"; \
    test -d "${NLTK_DATA}/tokenizers/punkt"; \
    test -d "${NLTK_DATA}/tokenizers/punkt_tab"; \
    test -f "${FASTTEXT_MODEL_ROOT}/cc.en.30.bin"; \
    test -f "${FASTTEXT_MODEL_ROOT}/cc.fr.30.bin"; \
    test -f "${WHISPER_MODEL_ROOT}/base.pt"; \
    test -d "${HF_HOME}/sentence-transformers/all-MiniLM-L12-v2"; \
    test -d "${HF_HOME}/Helsinki-NLP/opus-mt-tc-big-en-fr"; \
    test -d "${HF_HOME}/Helsinki-NLP/opus-mt-tc-big-fr-en"; \
    test -d "${HF_HOME}/Helsinki-NLP/opus-mt-de-en"; \
    test -d "${HF_HOME}/Helsinki-NLP/opus-mt-it-en"; \
    test -d "${HF_HOME}/Helsinki-NLP/opus-mt-en-de"; \
    test -d "${HF_HOME}/Helsinki-NLP/opus-mt-en-it"; \
    test -d "${HF_HOME}/Davlan/distilbert-base-multilingual-cased-ner-hrl"; \
    test -d "${HF_HOME}/urchade/gliner_multi_pii-v1"; \
    test -d "${HF_HOME}/microsoft/mdeberta-v3-base"; \
    python -c "import en_core_web_sm, fr_core_news_sm; print('spaCy models available')"

# Copy the entrypoint script, example configuration, and environment variable files to the container
COPY docker/entrypoint.sh /usr/local/bin/graphai-entrypoint.sh
COPY example-config.ini ./config.ini
COPY example-env ./.env

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/graphai-entrypoint.sh

# Copy the graphai package into the container
COPY graphai ./graphai

# Install the graphai package without dependencies since they are already installed.
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --no-deps .

# Expose the necessary ports for the GraphAI service
EXPOSE 28800 5555

# Set the entrypoint to use tini for proper signal handling and to run the graphai-entrypoint.sh script
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/graphai-entrypoint.sh"]
