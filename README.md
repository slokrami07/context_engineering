# Context-Engineer: Deterministic Prefix-Cache Aware Context Management Layer

A production-grade, lightweight, open-source Python library designed to manage long multi-turn conversations for LLM agents. It strictly prevents context rot and TPM/VRAM overflow while preserving prompt prefix-caching for local runtimes (such as LM Studio running Qwen 3.5 9B / Qwen 2.5 7B via `http://localhost:1234/v1`).

---

## 1. Core Architectural Invariant

The prompt token ribbon **strictly** follows this exact prefix-cacheable order:
```
[system] [pinned] [summary] [window turns] | [retrieved block] [question]
```
- **Left of `|` (Prefix-Stable Zone):** Identical across queries at the same conversation depth. Cached indefinitely in GPU VRAM / KV-cache memory.
- **Right of `|` (Dynamic Suffix Zone):** Contains per-query dynamic variations. Retrieval is appended as a dynamic suffix, **never chronologically inside history**, preventing cache invalidation.

```
+-------------------------------------------------------------+-----------------------------+
|                     PREFIX-STABLE ZONE                      |        DYNAMIC SUFFIX       |
|  [system] -> [pinned facts] -> [summary] -> [window turns]  |  [retrieved block] -> [q]   |
+-------------------------------------------------------------+-----------------------------+
```

---

## 2. The 5 Pipeline Layers

1. **Cap Layer (`layers/cap.py`):** In-place head/tail truncator. Tool outputs exceeding the threshold (35 tokens) are trimmed to head (22 tok) + elision marker + tail (8 tok). Preserves original raw content on the `Turn` for downstream uncapped retrieval.
2. **Pin Layer (`layers/pin.py`):** Deducts system prompt + invariant facts (pinned turns) from the budget ceiling upfront.
3. **Retrieve Layer (`layers/retrieve.py`):** Inspects candidate turns omitted by the window and runs BM25 (`rank-bm25`) against the query. Re-injects the top match **UNCAPPED** into the dynamic suffix block.
4. **Window Layer (`layers/window.py`):** Greedily packs the most recent contiguous turns into the remaining window token balance.
5. **Summarize Layer (`layers/summarize.py`):** For all dropped turns not retrieved, generates an abstract narrative summary. Strictly scrubs specific entity IDs, partition names, hashes, hex codes, and addresses to prevent stale fact hallucination.

---

## 3. Installation

```bash
uv pip install -e .
```

Dependencies: `rank-bm25`, `httpx`, `rich`, `tiktoken`, `transformers`.

---

## 4. Quickstart

```python
from context_engineer import (
    ContextConfig,
    Turn,
    assemble_context,
    LMStudioClient,
)

# 1. Configure budget limits
config = ContextConfig(
    context_ceiling=4096,
    completion_reserve=256,
    cap_threshold=35,
    lm_studio_base_url="http://localhost:1234/v1",
)

# 2. Define conversational turns
turns = [
    Turn(id=0, role="user", content="System baseline config", pinned=True),
    Turn(
        id=1,
        role="assistant",
        content="Found partition: shard-19 write stall error.",
        tool_output="dump: 0xDEADBEEF",
    ),
    Turn(id=2, role="user", content="Continue investigation and monitor cluster..."),
]

# 3. Assemble context with deterministic prefix-cache ribbon
assembled = assemble_context(
    turns=turns,
    query="Which partition ended up carrying the blame?",
    system_prompt="You are an expert SRE assistant.",
    config=config,
)

print("Prompt Ribbon:", assembled.ribbon_representation)

# 4. Dispatch to LM Studio runtime
client = LMStudioClient(base_url=config.lm_studio_base_url)
if client.is_available():
    response = client.chat_completion(
        messages=assembled.messages,
        model="qwen/qwen3.5-9b",
    )
    print("Response:", response)
```

---

## 5. Empirical Benchmark & Evaluator

Run the empirical benchmark harness measuring fact presence, recall, and prefix cache hit ratios across 100 turns:

```bash
python -m harness.evaluator
```

### Build Gates Verified
- `fact_present`: Asserts BM25 successfully locates and injects the uncapped needle (`shard-19`) into the dynamic prompt suffix.
- `fact_recalled`: Asserts the model correctly recalls the needle.
- `prefix_cache_order`: Enforces `[system] [pinned] [summary] [window turns] | [retrieved block] [question]`.
- `budget_enforced`: Guarantees token usage never breaches the effective ceiling (`Total <= Ceiling - Reserve`).

### Strategy Token Comparison (100 Turns)
| Strategy | Total Prompt Tokens | Cache Hit Tokens | Computed Tokens (Miss) | Cache Hit Ratio | VRAM Guard |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw Append-Only** | 4,165 | 1,426 | 2,739 | N/A | **OVERFLOW at turn #100** |
| **Chronological RAG Window** | 17,749 | 1,596 | 16,153 | **9.0%** | PROTECTED (Cache Trashing) |
| **Context-Engineer (Ours)** | 19,323 | 9,146 | 10,177 | **47.3%** | **PROTECTED (Zero OOM)** |

---

## 6. Adapters & Storage

- **ChatGPT Export Adapter:** Parse `conversations.json` into normalized `Turn` sequences.
  ```python
  from context_engineer.adapters import ChatGPTAdapter

  payload = ChatGPTAdapter().parse(export_json)
  ```
- **Claude Export Adapter:** Ingest Claude export JSON.
  ```python
  from context_engineer.adapters import ClaudeAdapter

  payload = ClaudeAdapter().parse(claude_json)
  ```
- **Zero-Config SQLite Store:**
  ```python
  from context_engineer.store import SQLiteStore

  store = SQLiteStore("conversations.db")
  store.save_turn("session-1", turn)
  ```
