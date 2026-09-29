"""Context-Engineer demonstration script."""

import sys

from benchmarks.scenarios import generate_scenario
from context_engineer import (
    ContextConfig,
    LMStudioClient,
    assemble_context,
)


def main() -> None:
    print("=================================================================")
    print(" Context-Engineer: Deterministic Prefix-Cache Aware Layer")
    print("=================================================================\n")

    config = ContextConfig(
        context_ceiling=4096,
        completion_reserve=256,
        cap_threshold=35,
        cap_head_tokens=22,
        cap_tail_tokens=8,
        lm_studio_base_url="http://localhost:1234/v1",
    )

    # 1. Generate 100-turn procedural scenario with needle
    scenario = generate_scenario(
        name="demo_scenario",
        domain="systems",
        total_turns=100,
        seed=42,
    )
    primary_needle = scenario.needles[0]
    print(
        f"[+] Loaded 100-turn procedural history. Planted needle in Turn #{primary_needle.turn_id}."
    )

    # 2. Assemble context using the pipeline
    assembled = assemble_context(
        turns=scenario.turns,
        query=primary_needle.query,
        system_prompt=scenario.system_prompt,
        config=config,
    )

    print(f"\n[Prompt Token Ribbon]:\n  {assembled.ribbon_representation}\n")
    print("Token Accounting:")
    print(f"  - System Reserve:       {assembled.budget_plan.system_tokens} tokens")
    print(f"  - Pinned Reserve:       {assembled.budget_plan.pinned_tokens} tokens")
    print(f"  - Summary Tokens:       {assembled.budget_plan.summary_tokens} tokens")
    print(
        f"  - Window Turns ({len(assembled.window_turns)}):     {assembled.budget_plan.window_tokens} tokens"
    )
    print(f"  - Retrieval (Uncapped): {assembled.budget_plan.retrieval_tokens} tokens")
    print(f"  - User Query:           {assembled.budget_plan.query_tokens} tokens")
    print(
        f"  - Total Used:           {assembled.budget_plan.total_used_tokens} / {assembled.budget_plan.effective_budget} tokens"
    )
    print(f"  - Budget Slack:         {assembled.budget_plan.remaining_tokens} tokens\n")

    # 3. Check LM Studio runtime
    client = LMStudioClient(base_url=config.lm_studio_base_url)
    if client.is_available():
        models = client.list_models()
        model_to_use = models[0] if models else "qwen/qwen3.5-9b"
        print(f"[+] Connected to LM Studio at {config.lm_studio_base_url}.")
        print(f"[+] Active loaded model: '{model_to_use}'. Running inference...")
        try:
            response = client.chat_completion(
                messages=assembled.messages,
                model=model_to_use,
                temperature=0.0,
                max_tokens=256,
            )
            print(f"\n[LM Studio Response ({model_to_use})]:\n{response}\n")
        except Exception as e:
            print(f"[!] Inference error: {e}")
    else:
        print(f"[*] LM Studio at {config.lm_studio_base_url} is offline.")
        print("    Running offline benchmark evaluation suite instead...\n")
        from benchmarks.run import main as run_benchmarks

        # Run with mock backend offline
        sys.argv = ["benchmarks.run", "--backend", "mock", "--turns", "100"]
        run_benchmarks()


if __name__ == "__main__":
    main()
