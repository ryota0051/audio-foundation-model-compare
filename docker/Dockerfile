FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    HF_HOME=/workspace/.hf_cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.18 /uv /uvx /bin/

WORKDIR /workspace

# 依存のみ先に入れてレイヤーキャッシュを効かせる
COPY pyproject.toml uv.lock* ./
RUN uv python install 3.12 && uv sync --no-install-project --python 3.12

COPY src ./src
RUN uv sync --python 3.12

CMD ["uv", "run", "pytest"]
