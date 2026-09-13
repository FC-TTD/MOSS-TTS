from __future__ import annotations

from argparse import Namespace
import os

import gradio as gr

from api.moss_formal import ATTN_IMPLEMENTATION, DEVICE, MODEL_PATH, app
from clis.moss_tts_app import build_demo


args = Namespace(
    model_path=os.getenv("MODEL_PATH", MODEL_PATH),
    device=os.getenv("DEVICE", DEVICE),
    attn_implementation=os.getenv("ATTN_IMPLEMENTATION", ATTN_IMPLEMENTATION),
)

# Fusion app: preserve the formal FastAPI service surface and mount the
# MOSS-TTS Gradio app at the root path, matching the old fusion entrypoint
# shape (api.fusion:app) while keeping /generate and /api/tts active.
demo = build_demo(args).queue(max_size=16, default_concurrency_limit=1)
app = gr.mount_gradio_app(app, demo, path="")
