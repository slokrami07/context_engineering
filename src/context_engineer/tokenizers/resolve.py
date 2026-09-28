"""Thread-safe tokenizer resolution, caching, and fallback management."""

import threading
import warnings

from context_engineer.errors import TokenizerError, TokenizerUnavailable
from context_engineer.tokenizers.approx import ApproxTokenizer
from context_engineer.tokenizers.base import Tokenizer

_TOKENIZER_CACHE: dict[str, Tokenizer] = {}
_CACHE_LOCK = threading.Lock()
_WARNED_TOKENIZERS: set[str] = set()


def resolve_tokenizer(
    spec: Tokenizer | str = "approx",
    *,
    strict: bool = False,
    allow_download: bool = False,
    trust_remote_code: bool = False,
) -> Tokenizer:
    """Resolves a tokenizer specification to a concrete Tokenizer instance.

    Args:
        spec: Instance of Tokenizer, or string spec ('approx', 'tiktoken:<encoding>', 'hf:<repo_or_path>').
        strict: If True, raises TokenizerError on any failure instead of falling back to ApproxTokenizer.
        allow_download: If True, allows network downloads for remote models (default: False for security - K1).
        trust_remote_code: If True, allows custom remote code execution (default: False for security - K1).

    Returns:
        A ready-to-use Tokenizer instance.
    """
    if isinstance(spec, Tokenizer):
        return spec

    spec_str = str(spec).strip()
    cache_key = f"{spec_str}:{strict}:{allow_download}:{trust_remote_code}"

    with _CACHE_LOCK:
        if cache_key in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[cache_key]

        tok: Tokenizer
        try:
            if spec_str == "approx":
                tok = ApproxTokenizer()
            elif spec_str.startswith("tiktoken:"):
                from context_engineer.tokenizers.tiktoken_ import TiktokenTokenizer

                enc_name = spec_str.split(":", 1)[1]
                tok = TiktokenTokenizer(enc_name)
            elif spec_str.startswith("hf:"):
                from context_engineer.tokenizers.hf import HFTokenizer

                repo = spec_str.split(":", 1)[1]
                tok = HFTokenizer(
                    repo,
                    local_files_only=not allow_download,
                    trust_remote_code=trust_remote_code,
                )
            elif "/" in spec_str or "qwen" in spec_str.lower() or "llama" in spec_str.lower():
                from context_engineer.tokenizers.hf import HFTokenizer

                tok = HFTokenizer(
                    spec_str,
                    local_files_only=not allow_download,
                    trust_remote_code=trust_remote_code,
                )
            else:
                raise TokenizerUnavailable(f"Unrecognized tokenizer specification: '{spec_str}'")

            # Invariant K2: ONLY cache successfully loaded tokenizers
            _TOKENIZER_CACHE[cache_key] = tok
            return tok

        except (TokenizerUnavailable, Exception) as e:
            if strict:
                raise TokenizerError(
                    f"Failed to resolve tokenizer '{spec_str}' in strict mode: {e}. "
                    "Ensure dependencies are installed ('pip install context-engineer[hf,tiktoken]') "
                    f"or pre-download the weights."
                ) from e

            if spec_str not in _WARNED_TOKENIZERS:
                warnings.warn(
                    f"Tokenizer '{spec_str}' is unavailable ({e}). Falling back to ApproxTokenizer. "
                    "Token counts and budget allocations will be approximate.",
                    UserWarning,
                    stacklevel=2,
                )
                _WARNED_TOKENIZERS.add(spec_str)
            # Invariant K2: Never cache fallback under original key
            return ApproxTokenizer(id="approx")
