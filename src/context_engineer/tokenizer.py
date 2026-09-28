"""Accurate token counting with ChatML overhead (<|im_start|>, <|im_end|>).

Supports HuggingFace AutoTokenizer (Qwen/Qwen2.5-7B-Instruct) with robust
fallback to tiktoken (cl100k_base/o200k_base) for deterministic offline operation.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Special ChatML overhead constants:
# <|im_start|>{role}\n -> 3 tokens
# {content}
# <|im_end|>\n -> 2 tokens
# Net message formatting overhead: 3 tokens per message (<|im_start|>, role, \n, <|im_end|>)
# Plus prompt priming (<|im_start|>assistant\n) = 3 tokens.
CHATML_MESSAGE_OVERHEAD = 4
CHATML_PROMPT_PRIMING = 3


class BaseTokenizerWrapper(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...
    def count(self, text: str) -> int: ...


class TiktokenFallbackWrapper:
    """Fast, deterministic local tokenizer using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))


class HFTokenizerWrapper:
    """HuggingFace AutoTokenizer wrapper with ChatML special token awareness."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        res = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(t) for t in res]

    def decode(self, tokens: list[int]) -> str:
        res = self._tokenizer.decode(tokens, skip_special_tokens=False)
        return str(res)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))


_TOKENIZER_CACHE: dict[str, BaseTokenizerWrapper] = {}


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> BaseTokenizerWrapper:
    """Retrieves or initializes the tokenizer for the given model name."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    # First attempt: Try HuggingFace AutoTokenizer if transformers is installed
    try:
        from transformers import AutoTokenizer

        # Use local_files_only first if cached, or load with timeout
        hf_tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False,
        )
        hf_wrapper = HFTokenizerWrapper(hf_tok)
        _TOKENIZER_CACHE[model_name] = hf_wrapper
        return hf_wrapper
    except Exception as e:
        logger.debug("HuggingFace tokenizer load skipped (%s), falling back to tiktoken", e)

    # Fallback: tiktoken
    try:
        tik_wrapper = TiktokenFallbackWrapper()
        _TOKENIZER_CACHE[model_name] = tik_wrapper
        return tik_wrapper
    except Exception as e:
        logger.warning("Tiktoken failed (%s), using character heuristic fallback", e)

        class CharFallbackWrapper:
            def encode(self, text: str) -> list[int]:
                return [ord(c) for c in text]

            def decode(self, tokens: list[int]) -> str:
                return "".join(chr(t) for t in tokens)

            def count(self, text: str) -> int:
                return max(1, len(text) // 4) if text else 0

        char_wrapper = CharFallbackWrapper()
        _TOKENIZER_CACHE[model_name] = char_wrapper
        return char_wrapper


def count_tokens(text: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> int:
    """Accurately counts raw string tokens."""
    if not text:
        return 0
    tokenizer = get_tokenizer(model_name)
    return tokenizer.count(text)


def count_message_tokens(
    role: str,
    content: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> int:
    """Counts tokens of a single ChatML formatted message including header and delimiter overhead.

    <|im_start|>{role}\n{content}<|im_end|>\n
    """
    raw_tokens = count_tokens(content, model_name=model_name)
    # Role label tokens: 'system'/'user'/'assistant' is typically 1 token
    role_tokens = count_tokens(role, model_name=model_name)
    # Overhead: <|im_start|> (1), \n (1), <|im_end|> (1), \n (1) = 4 tokens + role_tokens
    return raw_tokens + role_tokens + CHATML_MESSAGE_OVERHEAD


def count_turn_tokens(turn: Any, model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> int:
    """Counts tokens for an entire Turn object including content and tool_output."""
    text = turn.get_effective_text()
    return count_message_tokens(turn.role, text, model_name=model_name)


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    from_tail: bool = False,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> str:
    """Truncates text so that its token count does not exceed max_tokens.

    Args:
        text: Input string.
        max_tokens: Maximum token limit.
        from_tail: If True, keeps the end of the text instead of the beginning.
        model_name: Model identifier.
    """
    if max_tokens <= 0:
        return ""
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    sliced_tokens = tokens[-max_tokens:] if from_tail else tokens[:max_tokens]
    return tokenizer.decode(sliced_tokens)


def slice_head_tail(
    text: str,
    head_tokens: int,
    tail_tokens: int,
    marker_format: str = "[... {elided_count} tokens elided ...]",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> str:
    """Slices a text into head + elision marker + tail tokens.

    Args:
        text: Original text.
        head_tokens: Number of leading tokens to keep.
        tail_tokens: Number of trailing tokens to keep.
        marker_format: Format string accepting {elided_count}.
        model_name: Model identifier.
    """
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)
    total_tokens = len(tokens)

    if total_tokens <= (head_tokens + tail_tokens):
        return text

    head_part = tokenizer.decode(tokens[:head_tokens])
    tail_part = tokenizer.decode(tokens[-tail_tokens:])
    elided_count = total_tokens - (head_tokens + tail_tokens)
    marker = marker_format.format(elided_count=elided_count)

    return f"{head_part.rstrip()}\n{marker}\n{tail_part.lstrip()}"
