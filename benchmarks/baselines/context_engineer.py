"""Context-Engineer full pipeline baseline."""

from benchmarks.baselines.base import Baseline, BaselineResult
from context_engineer.config import ContextConfig
from context_engineer.errors import ContextBudgetError
from context_engineer.pipeline import assemble_context
from context_engineer.protocols import Retriever
from context_engineer.tokenizers import Tokenizer
from context_engineer.types import Turn


class ContextEngineerBaseline(Baseline):
    """Context-Engineer full pipeline baseline with fixed reserves and quantized boundary."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "Context-Engineer (Full)"

    def assemble(
        self,
        turns: list[Turn],
        query: str,
        system_prompt: str,
        config: ContextConfig,
        tok: Tokenizer,
    ) -> BaselineResult:
        try:
            assembled = assemble_context(
                turns=turns,
                query=query,
                system_prompt=system_prompt,
                config=config,
                tokenizer=tok,
                retriever=self.retriever,
            )
            total_tokens = sum(tok.count(m.content) for m in assembled.messages)
            return BaselineResult(
                name=self.name,
                messages=assembled.messages,
                prompt_tokens=total_tokens,
                overflow=False,
                retrieved_turns=list(assembled.retrieved_turns),
            )
        except ContextBudgetError:
            return BaselineResult(
                name=self.name,
                messages=[],
                prompt_tokens=config.effective_budget + 1,
                overflow=True,
            )
