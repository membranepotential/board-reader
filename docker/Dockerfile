# syntax=docker/dockerfile:1
# One image, parametrised by ${PACKAGE}. Each service (edge / inference / viewer)
# builds the same way but installs only its own workspace member + its deps, so
# the layer split holds at the container level too. Resolution uses the committed
# uv.lock, which already bakes in the 7-day seasoning cutoff from pyproject.toml.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm

# Pinned uv (matches the build backend constraint uv_build>=0.11.16,<0.12).
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/

ARG PACKAGE
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# libglib2.0-0 is the one runtime lib opencv-python-headless needs; harmless where
# opencv is absent (viewer).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# All member manifests are needed to resolve the workspace; sources to build them.
COPY pyproject.toml uv.lock ./
COPY packages ./packages

RUN uv sync --frozen --no-dev --package "${PACKAGE}"

# The concrete command is supplied per-service in compose.yaml.
