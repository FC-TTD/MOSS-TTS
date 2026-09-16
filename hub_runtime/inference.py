"""Native run_inference copied from formal 10273b9; narrow codec ownership.

MOSS generation's past_key_values is request-local. The CPU audio tokenizer's
streaming state is shared, so encode/conversation and decode are exclusive;
the language-model generation between them shares read-only weights concurrently.
Original generation parameters, dtype, output conversion and status are retained.
"""
import time
import numpy as np
import torch


def run_inference(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_tokens: int,
    language_tag: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    model_path: str,
    device: str,
    attn_implementation: str,
    max_new_tokens: int,
    *, backend, codec_lock, native,
):
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = backend
    duration_enabled = bool(duration_control_enabled and native.supports_duration_control(mode_with_reference))
    expected_tokens = int(duration_tokens) if duration_enabled else None
    with codec_lock:
        conversations, mode, mode_name = native.build_conversation(
            text=text,
            reference_audio=reference_audio,
            mode_with_reference=mode_with_reference,
            expected_tokens=expected_tokens,
            language_tag=language_tag,
            processor=processor,
        )

        batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )

    with codec_lock:
        messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise RuntimeError("模型没有返回可解码的音频结果。")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)

    elapsed = time.monotonic() - started_at
    normalized_language = native.normalize_language_tag(language_tag)
    status = (
        f"完成 | 模式：{mode_name} | 语言：{native.display_language_tag(normalized_language)} | "
        f"耗时：{elapsed:.2f}秒 | "
        f"max_new_tokens={int(max_new_tokens)}, "
        f"expected_tokens={expected_tokens if expected_tokens is not None else '关闭'}, "
        f"audio_temperature={float(temperature):.2f}, audio_top_p={float(top_p):.2f}, "
        f"audio_top_k={int(top_k)}, audio_repetition_penalty={float(repetition_penalty):.2f}"
    )
    return (sample_rate, audio_np), status

