"""Native formal UI constants/pure helpers; no model or UI imports."""
import re

DEFAULT_ATTN_IMPLEMENTATION = "auto"

DEFAULT_MAX_NEW_TOKENS = 4096

MODE_CLONE = "克隆音色"

MODE_CONTINUE = "续写"

MODE_CONTINUE_CLONE = "续写并克隆音色"

ZH_TOKENS_PER_CHAR = 3.098411951313033

EN_TOKENS_PER_CHAR = 0.8673376262755219

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

def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"

def estimate_duration_tokens(text: str) -> tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens
