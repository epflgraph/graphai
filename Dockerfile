FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
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

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /app

RUN apt-get update && \
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

COPY environment.yml ./environment.yml

ARG MICROMAMBA_VERSION=2.5.0-2
RUN curl -Ls https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64 \
    -o micromamba && \
    echo "c04571cfb0750e5432d530a3068b8fcd232ebed3133358e056e59a90b9852b00  micromamba" | sha256sum -c - && \
    install -m 0755 micromamba /usr/local/bin/micromamba && \
    rm micromamba && \
    micromamba create -y -n tools -f /app/environment.yml && \
    micromamba clean -a -y

COPY constraints.txt pyproject.toml setup.py README.md ./
COPY docker/requirements-base.txt docker/requirements-ml.txt ./docker/

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -c constraints.txt -r docker/requirements-base.txt -r docker/requirements-ml.txt

RUN mkdir -p \
      "${NLTK_DATA}" \
      "${GRAPH_CACHE_ROOT}" \
      "${GRAPH_LOG_ROOT}" \
      "${HF_HOME}" \
      "${WHISPER_MODEL_ROOT}" \
      "${FASTTEXT_MODEL_ROOT}" && \
    chmod -R 755 /opt/models /var/graphai

RUN python -m spacy download en_core_web_sm && \
    python -m spacy download fr_core_news_sm && \
    python -c "import os,nltk; target=os.environ.get('NLTK_DATA','/opt/nltk_data'); [nltk.download(p, download_dir=target, quiet=False) for p in ('stopwords','punkt','punkt_tab')]; print('NLTK data initialized at', target)"

COPY docker/preload_models.py /app/docker/preload_models.py

ARG WHISPER_MODEL_TYPE=base
ARG FASTTEXT_DIM=30
RUN python /app/docker/preload_models.py && \
    whisper --model "${WHISPER_MODEL_TYPE}" --download_root "${WHISPER_MODEL_ROOT}" --help >/dev/null 2>&1 && \
    fasttext-reduce --root_dir "${FASTTEXT_MODEL_ROOT}" --lang en --dim "${FASTTEXT_DIM}" && \
    fasttext-reduce --root_dir "${FASTTEXT_MODEL_ROOT}" --lang fr --dim "${FASTTEXT_DIM}"

COPY docker/entrypoint.sh /usr/local/bin/graphai-entrypoint.sh
COPY example-config.ini ./config.ini
COPY example-env ./.env

RUN chmod +x /usr/local/bin/graphai-entrypoint.sh

COPY graphai ./graphai

RUN python -m pip install --no-deps .

EXPOSE 28800 5555

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/graphai-entrypoint.sh"]
