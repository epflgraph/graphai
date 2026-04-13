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
    SENTENCE_TRANSFORMERS_HOME=/opt/models/huggingface \
    WHISPER_MODEL_ROOT=/opt/models/whisper \
    FASTTEXT_MODEL_ROOT=/opt/models/fasttext \
    PATH=/opt/micromamba/envs/tools/bin:$PATH

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

# Activate the tools environment for subsequent commands
COPY constraints.txt pyproject.toml setup.py ./

# Install Python dependencies
COPY docker/requirements-base.txt docker/requirements-ml.txt ./docker/

# Upgrade pip, setuptools, and wheel, then install the required Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -c constraints.txt -r docker/requirements-base.txt -r docker/requirements-ml.txt

# Second stage for models
FROM base AS models

# Create necessary directories and set permissions
RUN mkdir -p \
      "${NLTK_DATA}" \
      "${GRAPH_CACHE_ROOT}" \
      "${GRAPH_LOG_ROOT}" \
      "${HF_HOME}" \
      "${WHISPER_MODEL_ROOT}" \
      "${FASTTEXT_MODEL_ROOT}" && \
    chmod -R 755 /opt/models /var/graphai

# Copy the preload_models.py script and run it to download and prepare the necessary models for Whisper and FastText
COPY docker/preload_models.py /app/docker/preload_models.py

# Set build arguments for model dimensions
ARG FASTTEXT_DIM=30

# Download the necessary models for spaCy and NLTK, ensuring that the required language models and data are available for use in the application
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm
RUN python -c "import os,nltk; target=os.environ.get('NLTK_DATA','/opt/nltk_data'); [nltk.download(p, download_dir=target, quiet=False) for p in ('stopwords','punkt','punkt_tab')]; print('NLTK data initialized at', target)"

# Preload the models for sentence transformers, translation, and named entity recognition using the preload_models.py script
RUN python /app/docker/preload_models.py --group sentence
RUN python /app/docker/preload_models.py --group translation
RUN python /app/docker/preload_models.py --group ner

# Download and reduce the FastText models for English and French, then remove the original 300-dimension files in the same layer
RUN set -eux; \
    fasttext-reduce --root_dir "${FASTTEXT_MODEL_ROOT}" --lang en --dim "${FASTTEXT_DIM}"; \
    fasttext-reduce --root_dir "${FASTTEXT_MODEL_ROOT}" --lang fr --dim "${FASTTEXT_DIM}"; \
    rm -f \
      "${FASTTEXT_MODEL_ROOT}/cc.en.300.bin" \
      "${FASTTEXT_MODEL_ROOT}/cc.fr.300.bin"

# Copy the entrypoint script, example configuration, and environment variable files to the container
COPY docker/entrypoint.sh /usr/local/bin/graphai-entrypoint.sh
COPY example-config.ini ./config.ini
COPY example-env ./.env

# Make the entrypoint script executable
RUN chmod +x /usr/local/bin/graphai-entrypoint.sh

# Copy the graphai package into the container
COPY graphai ./graphai

# Install the graphai package without dependencies (since they are already installed) and ensure it's available in the PATH
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --no-deps .

# Expose the necessary ports for the GraphAI service
EXPOSE 28800 5555

# Set the entrypoint to use tini for proper signal handling and to run the graphai-entrypoint.sh script
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/graphai-entrypoint.sh"]
