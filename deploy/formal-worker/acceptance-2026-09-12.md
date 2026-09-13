# MOSS formal worker migration acceptance

Result: service migration completed. The original main-GPU/audio-tokenizer-CPU
design is preserved. The proposed all-GPU and dual-GPU layouts were withdrawn
before any runtime deployment after the user clarified the scope; no model
algorithm, precision, offload, guard or memory-limit change was deployed.

- Deployment source commit: `10273b9f6a31c3784c02201093c3366130b0bbbd`.
- Original and final image configuration ID: `sha256:f267ece35e105803ac7f925dacaa325dafedf743c7dd8b4a05d6994eb5913a88`.
- Published unchanged artifact: `registry.ttd/moss-tts/fusion:h-f267ece35e10`, pinned
  manifest digest `sha256:d32ae9314890ee0a2140a39cde4845453c6868c14c1388a4628e0b24cd1b0505`.
- Worker release: `/opt/moss-tts-formal/releases/10273b9f6a31c3784c02201093c3366130b0bbbd/`.
  All four active Python files match the source manifest and the former edge
  host-mounted code byte-for-byte. No new image build or package change occurred.
- Compose management path: `/opt/moss-tts-formal/compose.yaml`, project
  `moss-tts-formal`, service `moss-tts`, container `moss-tts-formal`.
- Main model `DEVICE=cuda:0`, worker GPU0 UUID
  `GPU-658f2323-d990-2adf-d488-d5dd945a27d5`; original
  `MOSS_AUDIO_TOKENIZER_DEVICE=cpu` remains unchanged. The existing worker driver
  was retained; this migration does not claim a new driver/DKMS installation.
- Shared weights/data remain on the same NFS source. Only the host prefix changed
  from edge `/mnt/TTD-Data` to worker `/TTD-Data`; runtime code mounts now point
  to the recorded release rather than an unversioned edge directory.

Candidate real API results (24kHz WAV): ordinary generation2.24s;
speed0.65 output3.938s; speed1.35 output1.7185s; expected_duration3.0 output3.0s;
reference-audio cloning3.52s/169004bytes. Existing `/model/unload` returned loaded
false and the attributable GPU process dropped to334MiB. This was tested only
on the isolated candidate, not during concurrent production traffic.

After promotion, `http://moss-tts/generate` returned a valid142124-byte,
24kHz/2.96s WAV, and `/model/status` reported loaded=true/device=cuda:0.
The Gradio `/config` returned200 with27 components. Effective Caddy configuration
contains only worker upstream `10.0.3.144:8000`, verified against its container
network. The final Python PID3919258 was attributed to worker GPU0 at16856MiB.
Worker MemAvailable was68065MiB; this is a snapshot, not a peak-memory guarantee.

The old edge container was already stopped at the start of the task. After
worker acceptance, full ID `9e217e163970a41cc5f6915119003fe7e232ebf66d660f511d75beb97ff44557`
was removed and its unchanged formal compose archived under
`/opt/openmoss-moss-tts/retired-formal-20260912T105902Z/`. The old image and data
remain available for explicit rollback. Other edge models and all stage/ASR and
P5000-v2 services were not selected by this migration.

Evidence: `/tmp/moss-worker-release.json`, `/tmp/moss-worker-candidate-evidence/`,
`/tmp/moss-worker-production-evidence/`, and `/tmp/moss-worker-promote.log` on the
control host. Rollback restores only the recorded edge formal definition and
stops the worker copy; it was not executed. No push to GitHub was performed.
