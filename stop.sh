#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-kolibri-1-vllm}"
PID_FILE="${SCRIPT_DIR}/.vllm.pid"
LOG_FILE="${SCRIPT_DIR}/.vllm.log"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
else
  echo "${CONTAINER_NAME} is not present"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
