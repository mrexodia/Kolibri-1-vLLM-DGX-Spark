#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Optional local overrides. Existing shell variables take precedence.
if [[ -f .env ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    [[ -n "${!key:-}" ]] || export "${key}=${value}"
  done < .env
fi

MODEL_ID="${MODEL_ID:-Aleph-Alpha/Kolibri-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kolibri-1}"
IMAGE="${IMAGE:-kolibri-1-gb10:local}"
CONTAINER_NAME="${CONTAINER_NAME:-kolibri-1-vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# Leave host headroom on the GB10's unified 128 GB memory. A 0.70 target
# still holds several full 262K KV sequences while leaving RAM for agents,
# Docker builds, and compilers. At 0.90 the host was close to swapping.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"
HOST_HF_HOME="${HOST_HF_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}}"
TRITON_CACHE_DIR="${SCRIPT_DIR}/.cache/triton"
VLLM_CACHE_DIR="${SCRIPT_DIR}/.cache/vllm"
LOG_FILE="${SCRIPT_DIR}/.vllm.log"
PID_FILE="${SCRIPT_DIR}/.vllm.pid"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH" >&2; exit 1; }
mkdir -p "${HOST_HF_HOME}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "${CONTAINER_NAME} is already running"
    echo "API: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Building ${IMAGE} (pinned vllm-gb10 0.29 + Aleph Alpha plugin)"
  docker build --pull -t "${IMAGE}" "${SCRIPT_DIR}"
fi

read -ra EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS:-}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching ${MODEL_ID}
EOF

echo "Starting ${MODEL_ID} as ${SERVED_MODEL_NAME}"
echo "Context=${MAX_MODEL_LEN}, max sequences=${MAX_NUM_SEQS}, GPU memory utilization=${GPU_MEMORY_UTILIZATION}"
echo "Hugging Face cache: ${HOST_HF_HOME}"
echo "Log: ${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -v "${HOST_HF_HOME}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE_DIR}:/root/.cache/triton" \
  -v "${VLLM_CACHE_DIR}:/root/.cache/vllm" \
  "${IMAGE}" \
  vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --reasoning-parser kolibri1 \
  --tool-call-parser kolibri1 \
  --enable-auto-tool-choice \
  "${EXTRA_ARGS_ARRAY[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
printf '%s\n' "${container_id}" >"${PID_FILE}"
(docker logs -f "${CONTAINER_NAME}" >>"${LOG_FILE}" 2>&1) &
log_pid=$!
trap 'kill "${log_pid}" 2>/dev/null || true' EXIT

echo "Waiting for ${READY_URL} (timeout ${STARTUP_TIMEOUT}s)"
deadline=$((SECONDS + STARTUP_TIMEOUT))
while ! curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container exited before readiness" >&2
    tail -n 200 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for readiness; container left running for inspection" >&2
    tail -n 100 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Kolibri is ready"
echo "OpenAI base URL: http://127.0.0.1:${PORT}/v1"
echo "Metrics: http://127.0.0.1:${PORT}/metrics"
