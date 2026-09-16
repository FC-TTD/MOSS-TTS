"""Copied formal clis/moss_tts_app.py UI; only GPU lifecycle/callback wiring changes."""
import argparse
import functools
import importlib.util
from pathlib import Path
import re
import time
import orjson
import os

import gradio as gr
from gradio_client import Client, handle_file
import numpy as np

MODEL_PATH = "OpenMOSS-Team/MOSS-TTS-v1.5"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 4096
CONTINUATION_NOTICE = (
    "续写模式已启用。请把参考音频的原文放在输入文本最前面。"
)

MODE_CLONE = "克隆音色"
MODE_CONTINUE = "续写"
MODE_CONTINUE_CLONE = "续写并克隆音色"
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219
REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = Path(__file__).resolve().parent.parent / "assets" / "text" / "moss_tts_example_texts.jsonl"
LANGUAGE_TAG_AUTO = "自动（不指定）"
LANGUAGE_TAG_MAP = {
    LANGUAGE_TAG_AUTO: None,
    "中文": "Chinese",
    "粤语": "Cantonese",
    "英语": "English",
    "阿拉伯语": "Arabic",
    "捷克语": "Czech",
    "丹麦语": "Danish",
    "荷兰语": "Dutch",
    "芬兰语": "Finnish",
    "法语": "French",
    "德语": "German",
    "希腊语": "Greek",
    "希伯来语": "Hebrew",
    "印地语": "Hindi",
    "匈牙利语": "Hungarian",
    "意大利语": "Italian",
    "日语": "Japanese",
    "韩语": "Korean",
    "马其顿语": "Macedonian",
    "马来语": "Malay",
    "波斯语": "Persian (Farsi)",
    "波兰语": "Polish",
    "葡萄牙语": "Portuguese",
    "罗马尼亚语": "Romanian",
    "俄语": "Russian",
    "西班牙语": "Spanish",
    "斯瓦希里语": "Swahili",
    "瑞典语": "Swedish",
    "塔加洛语": "Tagalog",
    "泰语": "Thai",
    "土耳其语": "Turkish",
    "越南语": "Vietnamese",
}
LANGUAGE_TAG_CHOICES = list(LANGUAGE_TAG_MAP)


def _parse_example_id(example_id: str) -> tuple[str, int] | None:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))


def _resolve_reference_audio_path(language: str, index: int) -> Path | None:
    stem_candidates = [f"reference_{language}_{index}"]
    for stem in stem_candidates:
        for ext in (".wav", ".mp3"):
            audio_path = REFERENCE_AUDIO_DIR / f"{stem}{ext}"
            if audio_path.exists():
                return audio_path
    return None


def build_example_rows() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    with open(EXAMPLE_TEXTS_JSONL_PATH, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            sample = orjson.loads(line)
            parsed = _parse_example_id(sample.get("id", ""))
            if parsed is None:
                continue

            language, index = parsed
            text = str(sample.get("text", "")).strip()
            audio_path = _resolve_reference_audio_path(language, index)
            if audio_path is None:
                continue

            rows.append((sample['role'], str(audio_path), text))

    return rows


EXAMPLE_ROWS = build_example_rows()
ASR_BACKEND_URL = os.environ.get("ASR_BACKEND_URL", "http://santi")


@functools.lru_cache(maxsize=1)
def get_asr_client() -> Client:
    return Client(ASR_BACKEND_URL)


def auto_fill_reference_transcript(
    reference_audio: str | None,
    current_text: str | None,
    mode_with_reference: str,
) -> str:
    current_text = (current_text or "").strip()
    if not reference_audio or mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE}:
        return current_text

    try:
        transcript = get_asr_client().predict(
            handle_file(reference_audio),
            api_name="/auto_asr",
        )
    except Exception as exc:
        print(f"[ASR] Failed to transcribe reference audio: {exc}", flush=True)
        return current_text

    transcript = str(transcript or "").strip()
    if not transcript:
        return current_text
    if current_text.startswith(transcript):
        return current_text
    if not current_text:
        return transcript
    return f"{transcript}\n{current_text}"






def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def supports_duration_control(mode_with_reference: str) -> bool:
    return mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE}


def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens


def update_duration_controls(
    enabled: bool,
    text: str,
    current_tokens: float | int | None,
    mode_with_reference: str,
):
    if not supports_duration_control(mode_with_reference):
        return (
            gr.update(visible=False),
            "续写模式下不能使用时长控制。",
            gr.update(value=False, interactive=False),
        )

    checkbox_update = gr.update(interactive=True)
    if not enabled:
        return gr.update(visible=False), "时长控制未启用。", checkbox_update

    language, default_tokens, min_tokens, max_tokens = estimate_duration_tokens(text)
    # Slider is initialized with value=1 as a placeholder; treat it as "unset"
    # so first-time estimation uses the computed default instead of clamping to min.
    if current_tokens is None or int(current_tokens) == 1:
        slider_value = default_tokens
    else:
        slider_value = int(current_tokens)
        slider_value = max(min_tokens, min(max_tokens, slider_value))

    language_label = "中文" if language == "zh" else "英语"
    hint = (
        f"时长控制已启用 | 检测语言：{language_label} | "
        f"建议值={default_tokens}，范围=[{min_tokens}, {max_tokens}]"
    )
    return (
        gr.update(
            visible=True,
            minimum=min_tokens,
            maximum=max_tokens,
            value=slider_value,
            step=1,
        ),
        hint,
        checkbox_update,
    )


def normalize_language_tag(language_tag: str | None) -> str | None:
    language_tag = (language_tag or "").strip()
    if not language_tag:
        return None
    if language_tag in LANGUAGE_TAG_MAP:
        return LANGUAGE_TAG_MAP[language_tag]
    return language_tag


def display_language_tag(language_tag: str | None) -> str:
    if not language_tag:
        return "自动"
    for label, value in LANGUAGE_TAG_MAP.items():
        if value == language_tag:
            return label
    return language_tag


def build_conversation(
    text: str,
    reference_audio: str | None,
    mode_with_reference: str,
    expected_tokens: int | None,
    language_tag: str | None,
    processor,
):
    text = (text or "").strip()
    if not text:
        raise ValueError("请输入需要合成的文本。")

    user_kwargs = {"text": text}
    normalized_language = normalize_language_tag(language_tag)
    if normalized_language is not None:
        user_kwargs["language"] = normalized_language
    if expected_tokens is not None:
        user_kwargs["tokens"] = int(expected_tokens)

    if not reference_audio:
        conversations = [[processor.build_user_message(**user_kwargs)]]
        return conversations, "generation", "直接生成"

    if mode_with_reference == MODE_CLONE:
        clone_kwargs = dict(user_kwargs)
        clone_kwargs["reference"] = [reference_audio]
        conversations = [[processor.build_user_message(**clone_kwargs)]]
        return conversations, "generation", MODE_CLONE

    if mode_with_reference == MODE_CONTINUE:
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[reference_audio]),
            ]
        ]
        return conversations, "continuation", MODE_CONTINUE

    continue_clone_kwargs = dict(user_kwargs)
    continue_clone_kwargs["reference"] = [reference_audio]
    conversations = [
        [
            processor.build_user_message(**continue_clone_kwargs),
            processor.build_assistant_message(audio_codes_list=[reference_audio]),
        ]
    ]
    return conversations, "continuation", MODE_CONTINUE_CLONE


def render_mode_hint(reference_audio: str | None, mode_with_reference: str):
    if not reference_audio:
        return "当前模式：**直接生成**（未上传参考音频）"
    if mode_with_reference == MODE_CLONE:
        return "当前模式：**克隆音色**（将从参考音频中克隆说话人音色）"
    return f"当前模式：**{mode_with_reference}**  \n> {CONTINUATION_NOTICE}"


def apply_example_selection(
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_tokens: int,
    evt: gr.SelectData,
):
    if evt is None or evt.index is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if isinstance(evt.index, (tuple, list)):
        row_idx = int(evt.index[0])
    else:
        row_idx = int(evt.index)

    if row_idx < 0 or row_idx >= len(EXAMPLE_ROWS):
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    _, audio_path, example_text = EXAMPLE_ROWS[row_idx]
    duration_slider_update, duration_hint, duration_checkbox_update = update_duration_controls(
        duration_control_enabled,
        example_text,
        duration_tokens,
        mode_with_reference,
    )
    return (
        audio_path,
        example_text,
        render_mode_hint(audio_path, mode_with_reference),
        duration_slider_update,
        duration_hint,
        duration_checkbox_update,
    )




def build_ui(runtime):
    from ttd_model_runtime.integrations.gradio import task

    @task(runtime)
    def managed_inference(text, reference_audio, mode_with_reference,
                          duration_control_enabled, duration_tokens, language_tag,
                          temperature, top_p, top_k, repetition_penalty, max_new_tokens):
        return runtime.get().infer(text, reference_audio, mode_with_reference,
                                   duration_control_enabled, duration_tokens, language_tag,
                                   temperature, top_p, top_k, repetition_penalty, max_new_tokens)

    custom_css = """
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #111418;
      --muted: #4d5562;
      --line: #e5e7eb;
      --accent: #0f766e;
    }
    .gradio-container {
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
      color: var(--ink);
    }
    .app-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
    }
    .app-title {
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 6px;
      letter-spacing: 0.2px;
    }
    .app-subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }
    #output_audio {
      padding-bottom: 12px;
      margin-bottom: 8px;
      overflow: hidden !important;
    }
    #output_audio > .wrap {
      overflow: hidden !important;
    }
    #output_audio audio {
      margin-bottom: 6px;
    }
    #run-btn {
      background: var(--accent);
      border: none;
    }
    """

    with gr.Blocks(title="MOSS-TTS 语音合成", css=custom_css) as demo:
        gr.Markdown(
            """
            <div class="app-card">
              <div class="app-title">MOSS-TTS v1.5</div>
              <div class="app-subtitle">支持直接生成、音色克隆、续写、续写并克隆音色、语言标签和行内停顿标记</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="文本",
                    lines=9,
                    placeholder="请输入要合成的文本。使用续写模式时，请把参考音频的原文放在最前面。",
                )
                reference_audio = gr.Audio(
                    label="参考音频（可选）",
                    type="filepath",
                )
                mode_with_reference = gr.Radio(
                    choices=[MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE],
                    value=MODE_CLONE,
                    label="参考音频模式",
                    info="未上传参考音频时，会自动使用直接生成。",
                )
                mode_hint = gr.Markdown(render_mode_hint(None, MODE_CLONE))
                language_tag = gr.Dropdown(
                    choices=LANGUAGE_TAG_CHOICES,
                    value=LANGUAGE_TAG_AUTO,
                    label="语言标签",
                    info="可选。已知输入语言时建议指定，尤其是中文和英语以外的语言。",
                )
                duration_control_enabled = gr.Checkbox(
                    value=False,
                    label="启用时长控制（预期音频 Token 数）",
                )
                duration_tokens = gr.Slider(
                    minimum=1,
                    maximum=2,
                    step=1,
                    value=1,
                    label="预期音频 Token 数",
                    visible=False,
                )
                duration_hint = gr.Markdown("时长控制未启用。")

                with gr.Accordion("采样参数（音频）", open=True):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.05,
                        value=1.7,
                        label="温度",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        step=0.01,
                        value=0.8,
                        label="Top-p",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=200,
                        step=1,
                        value=25,
                        label="Top-k",
                    )
                    repetition_penalty = gr.Slider(
                        minimum=0.8,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        label="重复惩罚",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=256,
                        maximum=8192,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                        label="最大新 Token 数",
                    )

                run_btn = gr.Button("生成语音", variant="primary", elem_id="run-btn")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="输出音频", type="numpy", elem_id="output_audio")
                status = gr.Textbox(label="状态", lines=4, interactive=False)
                examples_table = gr.Dataframe(
                    headers=["参考音色", "示例文本"],
                    value=[[name, text] for name, _, text in EXAMPLE_ROWS],
                    datatype=["str", "str"],
                    row_count=(len(EXAMPLE_ROWS), "fixed"),
                    col_count=(2, "fixed"),
                    interactive=False,
                    wrap=True,
                    label="示例（点击一行填入输入区）",
                )

        reference_audio.change(
            fn=render_mode_hint,
            inputs=[reference_audio, mode_with_reference],
            outputs=[mode_hint],
        )
        reference_audio.change(
            fn=auto_fill_reference_transcript,
            inputs=[reference_audio, text, mode_with_reference],
            outputs=[text],
        )
        mode_with_reference.change(
            fn=render_mode_hint,
            inputs=[reference_audio, mode_with_reference],
            outputs=[mode_hint],
        )
        mode_with_reference.change(
            fn=auto_fill_reference_transcript,
            inputs=[reference_audio, text, mode_with_reference],
            outputs=[text],
        )
        duration_control_enabled.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        text.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        mode_with_reference.change(
            fn=update_duration_controls,
            inputs=[duration_control_enabled, text, duration_tokens, mode_with_reference],
            outputs=[duration_tokens, duration_hint, duration_control_enabled],
        )
        examples_table.select(
            fn=apply_example_selection,
            inputs=[mode_with_reference, duration_control_enabled, duration_tokens],
            outputs=[
                reference_audio,
                text,
                mode_hint,
                duration_tokens,
                duration_hint,
                duration_control_enabled,
            ],
        )

        run_btn.click(
            fn=managed_inference,
            api_name="lambda",
            inputs=[
                text,
                reference_audio,
                mode_with_reference,
                duration_control_enabled,
                duration_tokens,
                language_tag,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
            ],
            outputs=[output_audio, status],
        )
    return demo.queue(max_size=16, default_concurrency_limit=1)




