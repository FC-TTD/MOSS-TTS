# MOSS managed runtime

2026-09-16: formally managed on worker; existing domains, native API/UI and Gateway acceptance passed. Current reservation: 20 GiB. Full evidence: Hub `docs/proposals/model-compute-pool/worker-expansion-2026-09-16.md`. Latest validated runtime source: `17a6494`; the original adoption source was `927389a`. Documentation commits do not change the deployed image.

Baseline: `10273b9` on `FC-TTD/MOSS-TTS`, matching the worker formal runtime overrides.
The original Hub adoption added `hub_runtime/` and test files while leaving the native source files unchanged.

- `deploy/formal-worker/runtime/api/moss_formal.py`: SHA-256 `a5311df81d3af002706af4a0198f441ffc209a6827731afe77f33ee36e101d00`.
- `deploy/formal-worker/runtime/clis/moss_tts_app.py`: SHA-256 `a30ba1a83096b79093f1977cdfe860d92eb97d8ddb721a2606afc3761cedb198`.

`api.py` copies the three original Form routes, upload aliases, defaults, response headers and speed/expected-duration postprocessing. Only the actual synchronous callback is managed; native CPU postprocessing stays with the response. `/model/unload` explicitly directs operators to Hub and never unloads weights beside a live lease. No old SmartModel, CUDA-health webhook or old utility-plugin lifecycle starts.

`ui.py` is the original UI copy with GPU imports/loader/inference/CLI removed. All controls, descriptions, examples, conditional duration controls and ASR assistance remain. The Hub UI sends reference audio to the current `asrpri-api` REST contract instead of the retired `santi` Gradio endpoint; failures are visible to the user instead of silently leaving continuation text empty. Its generation callback uses Runtime; the original Gradio API name `lambda` and queue limits are retained. `NativeUIRoutes` serves the original root and Gradio aliases via an internal `/__gradio__` mount, without moving the business domain/root.

The GPU loader imports the published `clis.moss_tts_app` only inside the engine interpreter. The final image must copy the baseline formal override into `/app/clis/moss_tts_app.py`; do not assume the base image contains that same revision. Keep native `/app/assets` accessible for the copied examples. The stable CPU host listens on 8000; private Runtime control remains the SDK endpoint.

The facade retains the complete native cached `(model, processor, device, sample_rate)` tuple for measurement and inference. The native main model uses lease-local `cuda:0` and bfloat16; the existing audio tokenizer stays on CPU. Release clears the native lru_cache and facade references before allocator cleanup and engine exit. No weights are moved to CPU by the integration.

CPU verification uses Python 3.12.13, Gradio 6.18.0, FastAPI 0.136.3 and NumPy 2.1.0. Six tests compare the formal OpenAPI/defaults and copied native UI controls/event names, the current ASR REST contract and visible failure path, all reference upload aliases, continuation parameters, real SoX timing, cache-release hooks, the actual private HTTP loader thread, root Gradio queue/download and one shared managed engine. The fixture imports no Torch and loads no weights; this is not GPU or production acceptance. Gradio keeps an idle heartbeat socket open; fixture teardown has a two-second HTTP grace after asserting zero activities and confirmed engine exit, without changing production teardown.

Run with the model root and `tests` plus Hub SDK `src` on PYTHONPATH:

```sh
PYTHONPATH=/path/to/ttd-hub/sdk/python/src:tests:. GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1 python -m unittest discover -s tests -p test_hub_runtime.py -v
```


Current concurrency boundary: `RequestBatcher` runs a single persistent native
thread with **max_batch_size=1, window_seconds=0**; `run_batch` rejects B>1.
Each HTTP caller keeps its own SDK activity while waiting/executing, but native
inference remains serial. Healthy weights stay resident after completion and
across idle periods; only pool pressure or explicit lifecycle control unloads.
Release waits for the native lane before clearing the lru and model references.

Real GPU testing rejected both direct multithreading (four requests took111s
versus21s sequential) and native multirow batching (B=2/4 failed to produce EOS
and emitted58.88s from the same text whose valid B=1 output is7.6s). No attention,
dtype, offload or native sampling algorithm was changed to conceal the issue.
Final12 requests all matched the correct7.6s waveform SHA256; after60s idle the
same engine served the next request in5.82s. Generic CPU queue fixtures are not
proof of native B>1 correctness. Final source is `17a6494`, current20GiB budget
and exact evidence are in Hub `runtime-concurrency-2026-09-16.md`.
