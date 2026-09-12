# Migrate the existing formal MOSS-TTS to worker

Scope: relocate `moss-tts-formal` to worker GPU0. Preserve the existing base image,
API, WebUI, weights, command, timing/postprocessing behavior and shared data.
No driver change, model optimization, CPU offload or added memory limit.

The existing edge service was already stopped when this migration began.
Its formal API and CLI were host-mounted code outside Git. This directory records
exact copies of the four active Python files, excluding backups and bytecode;
`source-manifest.json` proves their identity. They are migration artifacts for
the existing formal service, not a claim that upstream preview source implements
the same formal API. Do not rebuild the old mutable preview base.

One explicit placement correction is necessary for the user's no-CPU-inference
requirement: old `MOSS_AUDIO_TOKENIZER_DEVICE=cpu` becomes `cuda:0`. Both the main
model and audio tokenizer use worker GPU0. This may increase VRAM occupancy;
if the existing model cannot run there, stop and report instead of introducing
CPU/offload or changing inference behavior. Worker had one free 24GiB3090 and
about69GiB host RAM available at preparation time; these are snapshots, not peak
guarantees. SVC-v1 is being handled separately on worker GPU2.

## Image and source preparation

The preserved edge image ID is
`sha256:f267ece35e105803ac7f925dacaa325dafedf743c7dd8b4a05d6994eb5913a88`.
Publish that exact existing artifact directly from edge to the internal registry
under `registry.ttd/moss-tts/fusion:h-f267ece35e10`, without moving `latest`.
Record the registry's actual manifest digest and deploy by that digest.

Commit this migration directory before runtime writes. Export the directory
from that commit, verify each runtime file against `source-manifest.json`, and
copy only this small source release to
`worker:/opt/moss-tts-formal/releases/<commit>/`. Weights remain on the shared
NFS root, with `/mnt/TTD-Data` changed to worker's actual `/TTD-Data` mount.

Render environment for the committed compose:

```
MOSS_IMAGE=registry.ttd/moss-tts/fusion@sha256:<manifest-digest>
MOSS_RUNTIME_DIR=/opt/moss-tts-formal/releases/<commit>/runtime
MOSS_HOST=moss-tts-worker-validation
```

Use only the `moss-tts` service from this dedicated compose. Verify the target
port17870, required shared directories and the caddy network before creating it.
Candidate validation hostname is separate from the production `moss-tts` host.
The runtime keeps the existing host port17870 and restart policy.

## Cutover and acceptance

1. Start the candidate from the committed compose and immutable image.
2. Perform an actual short synthesis through `/generate`; validate WAV output,
   CUDA PID/UUID, main/tokenizer placement, and existing speed/duration behavior.
   Do not initialize a second model through an auxiliary `docker exec` process.
3. Change only `MOSS_HOST=moss-tts`, recreate this worker container to apply the
   formal discovery label, and recheck the API/WebUI and actual synthesis.
4. Remove the already-stopped old edge container by full ID, after checking its
   image and name. Archive its active formal compose so it cannot automatically
   recreate the old placement. Preserve weights, data and an explicit rollback
   copy; do not change any other model, caller, ASR or the P5000 SVC-v2 instance.
5. Verify one MOSS formal instance and one effective Caddy upstream on worker.

Rollback before cutover removes/stops only the worker candidate and leaves edge
data intact. If restoring the old edge deployment is requested, use its recorded
original compose and runtime files, noting its original CPU-tokenizer setting;
never silently re-enable that setting against the user's current requirement.
