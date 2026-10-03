# Kolibri 1 on one DGX Spark

A small Docker recipe for [Aleph-Alpha/Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1), using the GB10-native vLLM image and Aleph Alpha's official inference plugin.

## Real pi agent performance

These numbers come from **real, end-to-end pi coding-agent sessions**, not isolated synthetic prompts. Pi 0.85.1 worked through two llama.cpp bug-fixing tasks under Margin Eval, repeatedly inspecting files, editing code, compiling, running tests, and consuming tool results. The run used one request at a time, high reasoning, a 262,144-token context, and Kolibri's recommended sampling (`temperature=1.0`, `top_p=0.97`, `top_k=128`). It ran for 39m11s, made 118 model requests and 167 tool calls, and passed 1 of 2 cases (11/16 combined automated rubric points).

| Measurement | Result |
|---|---:|
| Computed prefill throughput | **~2,481 tok/s** |
| Effective prompt throughput, including cache hits | ~82,328 tok/s |
| Generation throughput | **~42.0 tok/s** |
| Mean inter-token latency | 23.83 ms/token |
| Mean time to first token | **0.765 s** |
| TTFT median / p90 histogram bounds | ≤0.75 s / ≤2.5 s |
| Prefix-cache hit rate | **96.99%** |
| Total prompt-token traffic | 6,553,102 |
| Cached prompt tokens | 6,355,584 |
| Locally computed prompt tokens | 197,518 |
| Generated tokens | 80,685 |
| Total token traffic (prompt + generated) | 6,633,787 |
| Tokens requiring model compute (uncached prompt + generated) | 278,203 |

The average request contained 55,535 prompt tokens: 53,861 came from the prefix cache and 1,674 required prefill compute; average output was 684 tokens. “Effective prompt throughput” counts cached prompt traffic, while “computed prefill throughput” is the meaningful rate for tokens that actually ran through prefill.

## Stack

- `ghcr.io/timothystewart6/vllm-gb10:v0.29.0-gb10.1`
- `aleph-alpha-inference==1.0.0`
- FP8 model weights and FP8 KV cache
- Native 262,144-token context
- Kolibri reasoning and tool-call parsers
- Prefix caching and per-request cached-token reporting
- OpenAI-compatible API and Prometheus `/metrics`
- 70% GPU-memory target, leaving unified-memory headroom for agents and builds

The base image is intentionally pinned. Aleph Alpha's plugin supports vLLM 0.29, whereas the current `vllm-gb10:latest` uses vLLM 0.30.

## Run

The model is approximately 79 GB. Downloading first is recommended:

```bash
hf download Aleph-Alpha/Kolibri-1
```

The launcher mounts `${HF_HOME:-~/.cache/huggingface}`, so the normal `hf download` cache is reused.

```bash
./start.sh
# API:     http://127.0.0.1:8888/v1
# metrics: http://127.0.0.1:8888/metrics

./stop.sh
```

The first start builds a tiny derivative image. Loading the 79 GB checkpoint took about 7.5 minutes in the initial DGX Spark test; later starts reuse the persisted compilation cache. Follow progress with:

```bash
tail -f .vllm.log
```

Optional overrides can be placed in `.env`; see `.env.example`.

## pi

Merge the provider in `pi-models.example.json` into `~/.pi/agent/models.json`, then select `vllm/kolibri-1` in pi. The configuration maps pi's thinking controls to Kolibri's chat-template arguments and enables streaming usage collection.

vLLM reports each request's cache hits as:

```text
usage.prompt_tokens_details.cached_tokens
```

Pi maps this to `usage.cacheRead`; uncached prompt tokens become `usage.input`, and generated tokens become `usage.output`. The launcher flag `--enable-prompt-tokens-details` is required for this reporting. Prefix caching itself is enabled separately with `--enable-prefix-caching`.

## Quick checks

```bash
curl -s http://127.0.0.1:8888/v1/models | jq
curl -s http://127.0.0.1:8888/metrics | head

curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kolibri-1",
    "messages": [{"role": "user", "content": "Write a Python hello-world program."}],
    "max_tokens": 128,
    "stream": false,
    "chat_template_kwargs": {"reasoning_effort": "low"}
  }' | jq
```

For a first single-Spark bring-up, keep the native 262K context. Although Kolibri supports extrapolation to 1M, it increases KV-cache and serving costs and should be qualified separately.

## Initial validation

Validated on one GB10 with the pinned image and the 3 October 2026 Kolibri snapshot:

- 73.55 GiB model memory; approximately 7.5 minutes to read weights.
- At `GPU_MEMORY_UTILIZATION=0.80`: 23.25 GiB KV cache and 1,937,896 KV tokens. The recipe now defaults to 0.70 to reserve more host RAM during coding-agent evals while retaining room for multiple full 262K KV sequences.
- Plain completion, separated reasoning, structured tool calling, streaming, `/metrics`, and clean shutdown/restart passed.
- Prefix-cache repeat reported `cached_tokens: 32` directly through vLLM.
- A repeated pi request reported `input: 59`, `cacheRead: 1456`, `output: 4`; a pi coding-agent request successfully called the `bash` tool.

vLLM warns that no model-specific `E=384,N=512` GB10 MoE tuning file exists, so the default MoE configuration may not be optimal. This affects performance tuning, not functional serving.
