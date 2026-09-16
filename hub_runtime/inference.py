"""Native batch inference derived from formal 10273b9; preserve row ownership.

MOSS generation's past_key_values is request-local. The CPU audio tokenizer's
streaming state is shared, so encode/conversation and decode are exclusive;
the language-model generation between them combines compatible requests.
Original generation parameters, dtype, output conversion and status are retained.
"""
import time
import numpy as np
import torch


def run_batch(requests, *, backend, codec_lock, native):
    """Native batch shapes: [B,T,n_vq+1] input and ordered B decoded messages.

    Processor owns left padding and attention masks. Shared CPU codec state is
    exclusive; a single native generate handles all compatible request rows.
    """
    if not requests:raise ValueError("empty MOSS inference batch")
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = backend
    conversations = []
    details = []
    batch_mode = None
    first = requests[0]
    if len(requests) > 1 and (first.group_key() is None or any(r.group_key()!=first.group_key() for r in requests)):
        raise ValueError("incompatible native batch requests")
    with codec_lock:
        for item in requests:
            enabled = bool(item.duration_control_enabled and native.supports_duration_control(item.mode_with_reference))
            expected_tokens = int(item.duration_tokens) if enabled else None
            conversation, mode, mode_name = native.build_conversation(
                text=item.text, reference_audio=item.reference_audio,
                mode_with_reference=item.mode_with_reference,
                expected_tokens=expected_tokens, language_tag=item.language_tag,
                processor=processor,
            )
            if len(conversation)!=1 or (batch_mode is not None and mode!=batch_mode):
                raise ValueError("native conversation batch/mode mismatch")
            batch_mode = mode
            conversations.extend(conversation)
            details.append((expected_tokens,mode_name))
        batch = processor(conversations, mode=batch_mode)
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=int(first.max_new_tokens),
            audio_temperature=float(first.temperature), audio_top_p=float(first.top_p),
            audio_top_k=int(first.top_k), audio_repetition_penalty=float(first.repetition_penalty),
        )
    if len(outputs)!=len(requests):raise RuntimeError("native generation result count mismatch")
    with codec_lock:
        messages = processor.decode(outputs)
        if len(messages)!=len(requests):raise RuntimeError("native decode result count mismatch")
        # Copy CPU output before releasing the codec lock, so a later batch
        # cannot mutate any decoder-owned workspace underlying this result.
        audio_results = []
        for message in messages:
            if message is None or not message.audio_codes_list:
                raise RuntimeError("模型没有返回可解码的音频结果。")
            audio = message.audio_codes_list[0]
            if isinstance(audio, torch.Tensor):audio_np=audio.detach().float().cpu().numpy()
            else:audio_np=np.asarray(audio,dtype=np.float32)
            if audio_np.ndim>1:audio_np=audio_np.reshape(-1)
            audio_results.append(audio_np.astype(np.float32,copy=True))
    elapsed = time.monotonic() - started_at
    results=[]
    for item,(expected_tokens,mode_name),audio in zip(requests,details,audio_results):
        language=native.normalize_language_tag(item.language_tag)
        status=(
            f"完成 | 模式：{mode_name} | 语言：{native.display_language_tag(language)} | "
            f"耗时：{elapsed:.2f}秒 | "
            f"max_new_tokens={int(item.max_new_tokens)}, "
            f"expected_tokens={expected_tokens if expected_tokens is not None else '关闭'}, "
            f"audio_temperature={float(item.temperature):.2f}, audio_top_p={float(item.top_p):.2f}, "
            f"audio_top_k={int(item.top_k)}, audio_repetition_penalty={float(item.repetition_penalty):.2f}"
        )
        results.append(((sample_rate,audio),status))
    return results
