"""Copied formal API routes/defaults/timing; model calls use one managed activity."""
from __future__ import annotations

from io import BytesIO
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from ttd_model_runtime.audio.speed import time_stretch_wav
from ttd_model_runtime import HubError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .parameters import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_MAX_NEW_TOKENS,
    LANGUAGE_TAG_MAP,
    MODE_CLONE,
    MODE_CONTINUE,
    MODE_CONTINUE_CLONE,
    estimate_duration_tokens,
)

logger = logging.getLogger("moss-tts-formal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("uvicorn.access").addFilter(
    lambda r: "/health" not in r.getMessage() and "/docs" not in r.getMessage()
)

MODEL_PATH = os.getenv("MODEL_PATH", "/models/MOSS-TTS-v1.5")
DEVICE = os.getenv("DEVICE", "cuda:0")
ATTN_IMPLEMENTATION = os.getenv("ATTN_IMPLEMENTATION", DEFAULT_ATTN_IMPLEMENTATION)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
MOSS_TOKENS_PER_SECOND = float(os.getenv("MOSS_TOKENS_PER_SECOND", "25"))


SPEED_LABELS: dict[str, float] = {
    "极慢": 0.65,
    "较慢": 0.8,
    "正常": 1.0,
    "较快": 1.2,
    "极快": 1.35,
}
LANGUAGE_CODES: dict[str, str] = {
    "auto": "自动（不指定）",
    "zh": "中文",
    "yue": "粤语",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "pt": "葡萄牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "tr": "土耳其语",
}
REFERENCE_MODES = {
    "clone": MODE_CLONE,
    "zero-shot": MODE_CLONE,
    "continue": MODE_CONTINUE,
    "continuation": MODE_CONTINUE,
    "continue-clone": MODE_CONTINUE_CLONE,
    "continuation-clone": MODE_CONTINUE_CLONE,
}


def _parse_speed(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 1.0
    if isinstance(value, (int, float)):
        speed = float(value)
    else:
        text = str(value).strip()
        speed = SPEED_LABELS.get(text, math.nan)
        if not math.isfinite(speed):
            try:
                speed = float(text)
            except ValueError as exc:
                raise ValueError(f"invalid speed: {value}") from exc
    if not 0.25 <= speed <= 4.0:
        raise ValueError("speed must be between 0.25 and 4.0")
    return speed


def _resolve_language(language: str | None) -> str:
    normalized = (language or "auto").strip()
    if normalized in LANGUAGE_CODES:
        return LANGUAGE_CODES[normalized]
    if normalized in LANGUAGE_TAG_MAP:
        return normalized
    return LANGUAGE_CODES.get("auto", "自动（不指定）")


def _resolve_reference_mode(mode: str, has_reference: bool) -> str:
    normalized = (mode or "generation").strip().lower()
    if not has_reference:
        return MODE_CLONE
    return REFERENCE_MODES.get(normalized, MODE_CLONE)


def _duration_tokens(text: str, enabled: bool) -> tuple[bool, int]:
    if enabled:
        _lang, base, _min_tokens, _max_tokens = estimate_duration_tokens(text)
        return True, base
    return False, 1


def _audio_duration_seconds(audio_np: np.ndarray, sample_rate: int) -> float:
    arr = np.asarray(audio_np, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if not sample_rate:
        return 0.0
    return float(len(arr)) / float(sample_rate)


def _apply_timing_control(
    audio_np: np.ndarray,
    sample_rate: int,
    speed: float,
    expected_duration: float | None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    if speed <= 0:
        raise ValueError("speed must be between 0.25 and 4.0")
    if expected_duration is not None and expected_duration < 0.1:
        raise ValueError("expected_duration must be >= 0.1")

    original_duration = _audio_duration_seconds(audio_np, sample_rate)
    final_speed_factor = float(speed)
    if expected_duration is not None:
        final_speed_factor = original_duration / float(expected_duration) if expected_duration else 1.0

    out = np.asarray(audio_np, dtype=np.float32)
    if abs(final_speed_factor - 1.0) > 1e-6:
        if time_stretch_wav is None:
            raise RuntimeError("SDK speed control is unavailable")
        try:
            out = time_stretch_wav(
                out,
                int(sample_rate),
                float(final_speed_factor),
                allow_passthrough_on_failure=False,
            )
        except TypeError as exc:
            if "allow_passthrough_on_failure" not in str(exc):
                raise
            out = time_stretch_wav(out, int(sample_rate), float(final_speed_factor))
        except FileNotFoundError:
            # Some formal images include ttd_fastapi_utils but not the SoX binary
            # it shells out to. Fall back to the in-image librosa implementation
            # rather than reverting to model duration-token control.
            import librosa

            out = librosa.effects.time_stretch(
                np.asarray(out, dtype=np.float32),
                rate=float(final_speed_factor),
            )

    final_duration = _audio_duration_seconds(out, sample_rate)
    return out, {
        "speed": float(speed),
        "expected_duration": expected_duration,
        "original_duration_seconds": original_duration,
        "final_duration_seconds": final_duration,
        "final_speed_factor": final_speed_factor,
    }


def _save_upload(upload: UploadFile | None) -> str | None:
    if upload is None:
        return None
    suffix = Path(upload.filename or "prompt.wav").suffix or ".wav"
    data = upload.file.read()
    if not data:
        return None
    tmp = tempfile.NamedTemporaryFile(prefix="moss-prompt-", suffix=suffix, delete=False)
    try:
        tmp.write(data)
        return tmp.name
    finally:
        tmp.close()


def _wav_bytes(audio_tuple: tuple[int, np.ndarray]) -> tuple[bytes, float, int]:
    sample_rate, audio_np = audio_tuple
    arr = np.asarray(audio_np, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    buf = BytesIO()
    sf.write(buf, arr, int(sample_rate), format="WAV")
    duration = float(len(arr) / float(sample_rate or 1))
    return buf.getvalue(), duration, int(sample_rate)



def build_api(runtime):
    app = FastAPI(
        title="MOSS-TTS Formal API",
        description="Formal MOSS-TTS v1.5 API for Dayan New Bage migration",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/api")
    async def api_info() -> dict[str, Any]:
        return {
            "service": "moss-tts-formal",
            "engine": "moss-tts-v15",
            "status": "ok",
            "routes": ["/", "/health", "/capabilities", "/languages", "/generate", "/api/tts", "/v1/audio/speech", "/model/status", "/model/unload"],
        }


    @app.get("/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "modes": ["generation", "clone", "continue", "continue-clone", "zero-shot"],
            "ip_choices": [],
            "supports_speed": True,
            "supports_expected_duration": True,
            "speed": {"min": 0.25, "max": 4.0, "default": 1.0, "labels": SPEED_LABELS},
            "expected_duration": {"min": 0.1, "unit": "seconds", "maps_to": "postprocess_time_stretch"},
            "languages": [{"code": code, "label": label} for code, label in LANGUAGE_CODES.items()],
        }


    @app.get("/languages")
    async def languages() -> dict[str, Any]:
        return {"languages": [{"code": code, "label": label} for code, label in LANGUAGE_CODES.items()]}


    @app.get("/model/status")
    def model_status() -> dict[str, Any]:
        state=runtime.status()
        return {"loaded":state.get("residency")=="ready", "last_error":None if state.get("healthy") else "runtime_unhealthy", "model_path":MODEL_PATH,"device":DEVICE,"managed":state}

    @app.post("/model/unload")
    def model_unload():
        raise HTTPException(status_code=409, detail="Managed by Hub: use the operator service drain endpoint")

    @app.post("/generate")
    @app.post("/api/tts")
    @app.post("/v1/audio/speech")
    @runtime.task
    def generate(
        text: str = Form(""),
        mode: str = Form("generation"),
        prompt_text: str | None = Form(None),
        language: str | None = Form("auto"),
        speed: str | None = Form(None),
        expected_duration: float | None = Form(None),
        duration_control_enabled: bool = Form(False),
        duration_tokens: int | None = Form(None),
        temperature: float = Form(1.7),
        top_p: float = Form(0.8),
        top_k: int = Form(25),
        repetition_penalty: float = Form(1.0),
        max_new_tokens: int = Form(MAX_NEW_TOKENS),
        response_format: str = Form("wav"),
        prompt_wav: UploadFile | None = File(None),
        reference_audio: UploadFile | None = File(None),
        prompt_speech: UploadFile | None = File(None),
        ref_audio: UploadFile | None = File(None),
    ):
        if response_format.lower() != "wav":
            raise HTTPException(status_code=400, detail=f"unsupported response_format: {response_format}")
        if not text.strip():
            raise HTTPException(status_code=400, detail="text is required")
        reference_upload = prompt_wav or reference_audio or prompt_speech or ref_audio
        reference_path = _save_upload(reference_upload)
        try:
            parsed_speed = _parse_speed(speed)
            if duration_tokens is not None:
                resolved_duration_enabled = True
                resolved_duration_tokens = max(1, int(duration_tokens))
            else:
                # `speed` and `expected_duration` are TTD post-generation timing
                # controls. They must not be translated into MOSS model
                # duration-tokens because that perturbs continuation conversations.
                resolved_duration_enabled, resolved_duration_tokens = _duration_tokens(
                    text,
                    bool(duration_control_enabled and expected_duration is None and parsed_speed == 1.0),
                )
            mode_with_reference = _resolve_reference_mode(mode, bool(reference_path))
            result, status = runtime.get().infer(
                text, reference_path, mode_with_reference,
                resolved_duration_enabled, resolved_duration_tokens,
                _resolve_language(language), float(temperature), float(top_p), int(top_k),
                float(repetition_penalty), int(max_new_tokens),
            )
            sample_rate, audio_np = result
            audio_np, timing_meta = _apply_timing_control(
                audio_np,
                int(sample_rate),
                parsed_speed,
                expected_duration,
            )
            result = (sample_rate, audio_np)
            audio_bytes, measured_duration, sample_rate = _wav_bytes(result)
            headers = {
                "Content-Disposition": "attachment; filename=moss-tts.wav",
                "X-Engine": "moss-tts-v15",
                "X-Mode": mode,
                "X-Speed": str(parsed_speed),
                "X-Expected-Duration": "" if expected_duration is None else str(expected_duration),
                "X-Duration-Tokens": str(resolved_duration_tokens if resolved_duration_enabled else ""),
                "X-Original-Duration": f"{float(timing_meta['original_duration_seconds'] or 0.0):.3f}",
                "X-Measured-Duration": f"{measured_duration:.3f}",
                "X-Final-Speed-Factor": f"{float(timing_meta['final_speed_factor'] or 1.0):.6f}",
                "X-Sample-Rate": str(sample_rate),
                "X-Inference-Status": "ok",
            }
            return Response(content=audio_bytes, media_type="audio/wav", headers=headers)
        except (HTTPException, HubError):
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("MOSS-TTS generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if reference_path:
                try:
                    Path(reference_path).unlink(missing_ok=True)
                except Exception:
                    pass

    return app
