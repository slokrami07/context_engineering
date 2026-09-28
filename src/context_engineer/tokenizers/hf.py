"""HuggingFace AutoTokenizer wrapper with secure offline defaults and chat template awareness."""

from collections.abc import Sequence
from typing import Any

from context_engineer.errors import TokenizerUnavailable
from context_engineer.tokenizers.base import Tokenizer
from context_engineer.types import Message


class HFTokenizer(Tokenizer):
    """HuggingFace AutoTokenizer wrapper enforcing offline security and Jinja chat template counting."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise TokenizerUnavailable(
                f"transformers library is required for HFTokenizer ({e}). "
                "Install with: pip install 'context-engineer[hf]'"
            ) from e

        self.id = f"hf:{model_name}"
        self.model_name = model_name

        try:
            self._tokenizer: Any = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            raise TokenizerUnavailable(
                f"Failed to load HuggingFace tokenizer '{model_name}' (local_files_only={local_files_only}): {e}. "
                f"To pre-download the tokenizer cache, run with allow_download=True or run: huggingface-cli download {model_name}"
            ) from e

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        # add_special_tokens=False ensures content strings containing <|im_end|> are treated as plain text (K7)
        res = self._tokenizer.encode(text, add_special_tokens=False)
        return [int(t) for t in res]

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return str(self._tokenizer.decode(ids, skip_special_tokens=False))

    def count_messages(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> int:
        # Use target model's real Jinja chat template if available (K8)
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(
            self._tokenizer, "chat_template", None
        ):
            chat = [{"role": m.role, "content": m.content} for m in messages]
            try:
                tokens = self._tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
                return len(tokens)
            except Exception:
                # Fallback to per-message counting if template execution fails
                pass

        # ChatML fallback counting
        total = 0
        for m in messages:
            role_tokens = len(self.encode(m.role))
            content_tokens = len(self.encode(m.content))
            total += role_tokens + content_tokens + 4
        if add_generation_prompt:
            total += 3
        return total
