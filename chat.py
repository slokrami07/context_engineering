"""Interactive Live Chat CLI for Context-Engineer + LM Studio.

Allows you to chat live with your locally running LM Studio model while
Context-Engineer manages the context budget, capping, pinning, and prefix-cache ribbon.
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from context_engineer import (
    ContextConfig,
    Turn,
    assemble_context,
    LMStudioClient,
)

console = Console()


def display_welcome(model_name: str, base_url: str, config: ContextConfig) -> None:
    content = (
        f"[bold green]Connected to LM Studio[/bold green] at [cyan]{base_url}[/cyan]\n"
        f"Active Model: [bold yellow]{model_name}[/bold yellow]\n"
        f"Context Ceiling: [magenta]{config.context_ceiling}[/magenta] tokens "
        f"(Effective budget: [magenta]{config.effective_budget}[/magenta] tokens)\n\n"
        "[bold]Special Commands:[/bold]\n"
        "  [cyan]/pin <fact>[/cyan]  - Pin an invariant fact (never evicted, stays at prompt head)\n"
        "  [cyan]/tool <text>[/cyan] - Simulate a large tool output (triggers Layer 1 head/tail capping)\n"
        "  [cyan]/ribbon[/cyan]      - Inspect current prompt token ribbon & assembled messages\n"
        "  [cyan]/clear[/cyan]       - Reset conversation history\n"
        "  [cyan]/exit[/cyan]        - Quit chat"
    )
    console.print(Panel(content, title="Context-Engineer Live Chat", border_style="cyan"))


def main() -> None:
    config = ContextConfig(
        context_ceiling=4096,
        completion_reserve=256,
        cap_threshold=35,
        cap_head_tokens=22,
        cap_tail_tokens=8,
        lm_studio_base_url="http://localhost:1234/v1",
    )

    client = LMStudioClient(base_url=config.lm_studio_base_url)

    if not client.is_available():
        console.print(
            Panel(
                "[bold red]Cannot connect to LM Studio Local Server[/bold red]\n\n"
                f"Failed to reach [cyan]{config.lm_studio_base_url}[/cyan].\n\n"
                "[bold]How to start LM Studio Local Server:[/bold]\n"
                "1. Open [bold]LM Studio[/bold].\n"
                "2. Click the [bold]<-> (Local Server)[/bold] icon in the left sidebar.\n"
                "3. Select and load any downloaded model at the top.\n"
                "4. Click [bold]Start Server[/bold] (Port 1234).\n"
                "5. Re-run this script: [bold green]python chat.py[/bold green]",
                title="LM Studio Offline",
                border_style="red",
            )
        )
        sys.exit(1)

    models = client.list_models()
    model_name = models[0] if models else "default-model"

    display_welcome(model_name=model_name, base_url=config.lm_studio_base_url, config=config)

    system_prompt = (
        "You are an expert AI assistant. Provide concise, direct, and factually accurate responses."
    )
    turns: list[Turn] = []
    turn_counter = 0

    while True:
        try:
            user_input = console.input("\n[bold green]You > [/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/clear":
            turns.clear()
            turn_counter = 0
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue

        if user_input.startswith("/pin "):
            fact = user_input[5:].strip()
            pinned_turn = Turn(
                id=turn_counter,
                role="user",
                content=f"[Pinned Invariant Fact]: {fact}",
                pinned=True,
            )
            turns.append(pinned_turn)
            turn_counter += 1
            console.print(f"[bold cyan][Pinned Turn #{pinned_turn.id} Added]:[/bold cyan] {fact}")
            continue

        if user_input.startswith("/tool "):
            raw_tool = user_input[6:].strip()
            tool_turn = Turn(
                id=turn_counter,
                role="assistant",
                content="Executed diagnostic tool operation.",
                tool_output=raw_tool,
                pinned=False,
            )
            turns.append(tool_turn)
            turn_counter += 1
            console.print(f"[bold magenta][Tool Turn #{tool_turn.id} Added]:[/bold magenta] {len(raw_tool.split())} words")
            continue

        if user_input.lower() == "/ribbon":
            if not turns:
                console.print("[dim]No turns in history yet.[/dim]")
                continue
            dummy_assembled = assemble_context(
                turns=turns,
                query="<inspect>",
                system_prompt=system_prompt,
                config=config,
            )
            table = Table(title="Current Token Ribbon Accounting", border_style="blue")
            table.add_column("Layer Component", style="bold")
            table.add_column("Tokens", justify="right")
            bp = dummy_assembled.budget_plan
            table.add_row("System Reserve", f"{bp.system_tokens}t")
            table.add_row("Pinned Facts", f"{bp.pinned_tokens}t")
            table.add_row("Historical Summary", f"{bp.summary_tokens}t")
            table.add_row(f"Window Turns ({len(dummy_assembled.window_turns)})", f"{bp.window_tokens}t")
            table.add_row("Dynamic Retrieval (Uncapped)", f"{bp.retrieval_tokens}t")
            table.add_row("Total Used / Budget", f"{bp.total_used_tokens} / {bp.effective_budget}t")
            console.print(table)
            console.print(f"[dim]Ribbon:[/dim] [cyan]{dummy_assembled.ribbon_representation}[/cyan]")
            continue

        # 1. Assemble context through Context-Engineer
        assembled = assemble_context(
            turns=turns,
            query=user_input,
            system_prompt=system_prompt,
            config=config,
        )

        # 2. Display the Prompt Token Ribbon
        console.print(f"[dim cyan]Ribbon: {assembled.ribbon_representation}[/dim cyan]")

        # 3. Dispatch to LM Studio
        console.print("[dim]Generating response from LM Studio...[/dim]", end="\r")
        try:
            response = client.chat_completion(
                messages=assembled.messages,
                model=model_name,
                temperature=0.7,
                max_tokens=config.completion_reserve,
            )
        except Exception as e:
            console.print(f"\n[bold red][!] LM Studio Inference Error:[/bold red] {e}")
            continue

        # Clear line
        console.print(" " * 50, end="\r")
        console.print(f"[bold blue]{model_name} >[/bold blue] {response}")

        # 4. Save turns into history
        user_turn = Turn(id=turn_counter, role="user", content=user_input)
        assistant_turn = Turn(id=turn_counter + 1, role="assistant", content=response)
        turns.append(user_turn)
        turns.append(assistant_turn)
        turn_counter += 2


if __name__ == "__main__":
    main()
