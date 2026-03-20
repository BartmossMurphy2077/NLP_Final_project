"""Text transforms for masking and ablation studies."""

from __future__ import annotations

import re


EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\u2600-\u26ff"
    "\u2700-\u27bf"
    "]"
)

PROFANITY_WORDS = {
    "damn",
    "hell",
    "shit",
    "fuck",
    "fucking",
    "crap",
    "bitch",
    "ass",
}

WORD_RE = re.compile(r"\b[\w']+\b")


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub("", text)


def remove_profanity(text: str, replacement: str = "[PROFANITY]") -> str:
    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.lower() in PROFANITY_WORDS:
            return replacement
        return token

    return WORD_RE.sub(_repl, text)


def apply_ablation(
    text: str, remove_profanity_flag: bool, remove_emojis_flag: bool
) -> str:
    out = text
    if remove_profanity_flag:
        out = remove_profanity(out)
    if remove_emojis_flag:
        out = remove_emojis(out)
    return " ".join(out.split())


def apply_masking_mode(text: str, mode: str, mask_token: str) -> str:
    # The variants are already created upstream in preprocessing; this function is used
    # when additional runtime transforms are needed for experiments.
    if mode == "none":
        return text
    if mode in {"slang_masked", "mixed", "original"}:
        return text.replace("[SLANG]", mask_token)
    raise ValueError(f"Unsupported masking mode: {mode}")
