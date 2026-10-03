# Kolibri's plugin supports vLLM 0.29.x. Keep this GB10 image pinned;
# vllm-gb10:latest may move to an incompatible vLLM minor release.
FROM ghcr.io/timothystewart6/vllm-gb10:v0.29.0-gb10.1

RUN python3 -m pip install --no-cache-dir --no-deps \
    aleph-alpha-inference==1.0.0
