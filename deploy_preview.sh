#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the SSH destination}"
REMOTE_DIR="${REMOTE_DIR:-/opt/moss-tts-preview}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/var/lib/moss-tts-preview}"
COMPOSE_FILE="docker/compose.preview.yaml"

echo "[1/4] prepare remote directories on ${REMOTE_HOST}"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}' '${REMOTE_DATA_DIR}/hf' '${REMOTE_DATA_DIR}/outputs'"

echo "[2/4] sync repository to ${REMOTE_HOST}:${REMOTE_DIR}"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  "${SCRIPT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "[3/4] build image with classic docker builder and launch preview container"
ssh "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && DOCKER_BUILDKIT=0 docker build -t moss-tts-preview:local -f docker/Dockerfile.preview . && MOSS_TTS_DATA_DIR='${REMOTE_DATA_DIR}' docker compose -f '${COMPOSE_FILE}' up -d --no-build --force-recreate"

echo "[4/4] current service status"
ssh "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && MOSS_TTS_DATA_DIR='${REMOTE_DATA_DIR}' docker compose -f '${COMPOSE_FILE}' ps && docker logs --tail 60 moss-tts-preview || true"
