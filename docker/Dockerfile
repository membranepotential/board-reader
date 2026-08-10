# syntax=docker/dockerfile:1
# One image, parametrised by ${PACKAGE}. Each service (edge / inference / viewer)
# builds the same way but installs only its own workspace member + its deps, so
# the layer split holds at the container level too. Resolution uses the committed
# uv.lock, which already bakes in the 7-day seasoning cutoff from pyproject.toml.
#
# The viewer additionally ships a prerendered SvelteKit bundle (packages/viewer/ui)
# as package data (pyproject.toml's uv_build backend includes everything under
# src/board_reader_viewer/, web/ included — see packages/viewer/pyproject.toml). The
# `ui-build` stage below produces it; only `final-board-reader-viewer` pulls from it.
# BuildKit's build graph only executes the stages the selected target actually
# depends on, so `--build-arg PACKAGE=board-reader-edge` (or inference) never touches
# Node at all — those two images build exactly as fast, and stay exactly as free of
# JS tooling, as before this stage existed.
ARG PACKAGE
ARG PYTHON_VERSION=3.12
# Current Node LTS ("Krypton") as of writing, on the same Debian release
# (bookworm) as the Python base below — one apt family for both stages.
ARG NODE_VERSION=24.19.0
# Pinned to the pnpm already used to produce packages/viewer/ui/pnpm-lock.yaml
# (lockfileVersion 9.0), installed explicitly rather than via corepack: the UI
# package.json carries no `packageManager` field for corepack to read, and this repo
# does not own that file.
ARG PNPM_VERSION=11.17.0

FROM node:${NODE_VERSION}-bookworm-slim AS ui-build
WORKDIR /app
# `.dockerignore` already keeps the host's `node_modules/` and `.svelte-kit/` out of the
# build context. This is belt and braces: a stale local `.dockerignore`, or a context
# assembled some other way, would otherwise hand pnpm a foreign store built for the host
# — which it refuses to clobber without a TTY or `CI=true`, and whose native binaries are
# the wrong architecture anyway. Cheap to keep, and it fails loudly if the assumption
# above ever stops holding.
COPY packages/viewer/ui ./packages/viewer/ui
RUN rm -rf packages/viewer/ui/node_modules packages/viewer/ui/.svelte-kit
WORKDIR /app/packages/viewer/ui
ARG PNPM_VERSION
ENV CI=true
RUN npm install --global "pnpm@${PNPM_VERSION}" \
    && pnpm install --frozen-lockfile \
    && pnpm build
# adapter-static (svelte.config.js) writes straight into the Python package:
#   /app/packages/viewer/src/board_reader_viewer/web

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Pinned uv (matches the build backend constraint uv_build>=0.11.16,<0.12).
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /usr/local/bin/

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

# Per-package forks of `base`, selected by name below. Only the viewer's overlays the
# built UI — edge and inference are untouched copies of `base`.
FROM base AS final-board-reader-edge
FROM base AS final-board-reader-inference
FROM base AS final-board-reader-viewer
COPY --from=ui-build /app/packages/viewer/src/board_reader_viewer/web ./packages/viewer/src/board_reader_viewer/web

FROM final-${PACKAGE} AS final
ARG PACKAGE
RUN uv sync --frozen --no-dev --package "${PACKAGE}"

# The concrete command is supplied per-service in compose.yaml.
