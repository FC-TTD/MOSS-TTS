"""Use the published native MOSS loader inside the GPU process only."""
import gc
import os
import threading


def actual_tensor_placement(module):
    """Observe existing parameters/buffers; never infer placement from env."""
    for source in ("parameters", "buffers"):
        try:
            tensor = next(iter(getattr(module, source)()))
            return str(tensor.device), str(tensor.dtype)
        except Exception:
            continue
    return "unknown", "unknown"


class NativeBackend:
    def __init__(self, native, backend, model_path, device, attention):
        # Keep the full tuple visible to SDK tensor measurement, including the
        # native CPU audio tokenizer. No implicit .cpu(), dtype or map changes.
        self.backend = backend
        self.native = native
        self.model_path = model_path
        self.device = device
        self.attention = attention
        self._codec_lock = threading.RLock()
        main_device, main_dtype = actual_tensor_placement(backend[0])
        audio_device, audio_dtype = actual_tensor_placement(getattr(backend[1], "audio_tokenizer", None))
        self.__hub_device_summary__ = {
            "main_model": main_device,
            "main_dtype": main_dtype,
            "audio_tokenizer": audio_device,
            "audio_tokenizer_dtype": audio_dtype,
        }

    def infer(self, text, reference_audio, mode_with_reference,
              duration_control_enabled, duration_tokens, language_tag,
              temperature, top_p, top_k, repetition_penalty, max_new_tokens):
        from .inference import run_inference
        return run_inference(
            text, reference_audio, mode_with_reference,
            duration_control_enabled, duration_tokens, language_tag,
            temperature, top_p, top_k, repetition_penalty,
            self.model_path, self.device, self.attention, max_new_tokens,
            backend=self.backend, codec_lock=self._codec_lock, native=self.native,
        )


def load_model():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Managed MOSS requires its assigned CUDA GPU")
    device = os.getenv("DEVICE", "cuda:0")
    if device not in ("cuda", "cuda:0"):
        raise RuntimeError("Native MOSS main model must use the lease GPU as cuda:0")
    # This is the existing formal CPU/GPU split, not a capacity fallback.
    os.environ.setdefault("MOSS_AUDIO_TOKENIZER_DEVICE", "cpu")
    if os.environ["MOSS_AUDIO_TOKENIZER_DEVICE"] != "cpu":
        raise RuntimeError("Native formal audio tokenizer placement is CPU")
    from clis import moss_tts_app as native
    model_path = os.getenv("MODEL_PATH", "/models/MOSS-TTS-v1.5")
    attention = os.getenv("ATTN_IMPLEMENTATION", native.DEFAULT_ATTN_IMPLEMENTATION)
    backend = native.load_backend(model_path=model_path, device_str=device,
                                  attn_implementation=attention)
    return NativeBackend(native, backend, model_path, device, attention)


def completion(model):
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release(model):
    # Native lru_cache otherwise retains both model and processor after the
    # facade is dropped. Clear it before SDK releases the facade/allocator.
    if model.native is not None:
        model.native.load_backend.cache_clear()
    model.backend = None
    model.native = None


def cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
