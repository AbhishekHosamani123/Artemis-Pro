# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``artemis laya`` — status and self-test for the Laya decision engine.

Shows which backend is active (Laya server / OpenAI-compatible / lexical),
probes the configured Laya server, and optionally runs a relevance
benchmark against the live backend.
"""

from __future__ import annotations

import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

#: (query, label, expected) triples exercising exact, lexical-hard and
#: paraphrase-only matching — the cases that decide whether a target is
#: found locally or costs an extra VLM grounding round.
_BENCHMARK_PAIRS = [
    ("settings", "Settings", "exact"),
    ("search messages", "Search", "lexical"),
    ("save battery", "Battery saver", "lexical (word order)"),
    ("battery percentage", "Battery 85%", "lexical"),
    ("power options", "Battery", "paraphrase (semantic only)"),
    ("internet connection", "Network & internet", "paraphrase (semantic only)"),
    ("delete account", "Add account", "negative control"),
    ("battery status", "Volume: 60%", "negative control"),
]


def laya_command(
    benchmark: bool = typer.Option(
        False, "--benchmark", "-b", help="Run a relevance benchmark against the active backend."
    ),
) -> None:
    """Show Laya decision-engine status; with --benchmark, score sample pairs."""
    from artemis.decisions.engine import get_decision_engine, reset_decision_engine

    reset_decision_engine()
    engine = get_decision_engine()
    info = engine.info()

    console.print(
        Panel(
            "[bold cyan]Laya Decision Engine[/bold cyan] — millisecond relevance"
            " scoring for element search & history recall",
            title="☕ artemis laya",
            expand=False,
        )
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Backend mode", info["mode"])
    table.add_row("Active backend", f"[bold]{info['backend']}[/bold]" + (" (semantic)" if info["semantic"] else " (lexical only)"))
    table.add_row("Min confidence", f"{info['min_confidence']:.2f}")
    table.add_row("Rescue limit", str(info["rescue_limit"]))
    laya_state = "[red]not running[/red]"
    if info["laya"]:
        healthy, detail = info["laya"]
        laya_state = f"[green]{detail}[/green]" if healthy else f"[yellow]{detail}[/yellow]"
    elif info["mode"] not in ("openai",):
        laya_state = f"[dim]not probed (backend={info['backend']})[/dim]"
    table.add_row("Laya server", laya_state)
    table.add_row("Laya URL", info["laya_url"])
    table.add_row(
        "OpenAI-compatible endpoint",
        (f"configured → {info['openai_base_url']} (model {info['openai_model'] or 'default'})")
        if info["openai_configured"]
        else "no key",
    )
    console.print(table)

    if info["mode"] in ("auto", "lexical", "off") and not info["laya"]:
        console.print(
            "\n[dim]To enable true semantic matching, run a Laya server"
            " (Apple Silicon: [bold]pip install 'laya-mlx[serve]' && laya-mlx-serve[/bold],"
            " see github.com/mizorewww/laya-mlx; set ARTEMIS_LAYA_URL if remote) or opt into"
            " the OpenAI-compatible backend with"
            " [bold]ARTEMIS_DECISION_BACKEND=openai[/bold].[/dim]"
        )

    if not benchmark:
        return

    console.print()
    scores_table = Table(title="Relevance benchmark", expand=False)
    scores_table.add_column("Query", style="cyan")
    scores_table.add_column("Label")
    scores_table.add_column("Case")
    scores_table.add_column("Lexical", justify="right")
    scores_table.add_column("Engine", justify="right")

    from artemis.decisions.lexical import lexical_similarity

    started = time.perf_counter()
    for query, label, case in _BENCHMARK_PAIRS:
        lexical_score = lexical_similarity(query, label)
        engine_score = engine.rank(query, [label])[0]
        engine_text = f"{engine_score:.2f}" if engine.semantic else "—"
        scores_table.add_row(query, label, case, f"{lexical_score:.2f}", engine_text)
    elapsed_ms = (time.perf_counter() - started) * 1000

    console.print(scores_table)
    console.print(f"\n[green]{len(_BENCHMARK_PAIRS)} pairs scored in {elapsed_ms:.0f} ms"
                  f" via backend [bold]{info['backend']}[/bold].[/green]")
