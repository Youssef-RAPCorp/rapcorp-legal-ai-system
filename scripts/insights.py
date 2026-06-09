"""
CLI insight tool — ask questions in six different modes.

Usage:
    python scripts/insights.py
        Interactive REPL.

    python scripts/insights.py --mode evaluate "Should I file a motion to dismiss?"
        One-shot query.

    python scripts/insights.py --mode critique --context my_case.txt "Plan to ..."
        With a file of case context.

    python scripts/insights.py --list-modes
        Show all available modes.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make project root importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import LegalAIConfig
from src.insights import InsightEngine, InsightMode


_MODE_HELP = {
    "evaluate":   "Is this a good idea? Pros/cons + recommendation",
    "elaborate":  "Expand on the idea with concrete detail",
    "critique":   "Find risks, blind spots, failure modes",
    "brainstorm": "Generate alternative approaches/options",
    "explain":    "Break down a statute, case, concept, or doctrine",
    "open":       "Free-form question",
}


def _print_modes() -> None:
    print("\nAvailable insight modes:")
    for name, desc in _MODE_HELP.items():
        print(f"  {name:<11} — {desc}")
    print()


def _parse_mode(raw: str) -> InsightMode:
    try:
        return InsightMode(raw.lower().strip())
    except ValueError:
        print(f"Unknown mode: {raw!r}")
        _print_modes()
        sys.exit(2)


def _read_context_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"Context file not found: {path}")
        sys.exit(2)
    return p.read_text(encoding="utf-8", errors="replace")


def _print_response(resp) -> None:
    print()
    print("═" * 72)
    print(f"Mode: {resp.mode.value}")
    if resp.error:
        print(f"\nError: {resp.error}")
        return
    if resp.context_summary:
        print(f"Context: {resp.context_summary}")
    print("═" * 72)
    print()
    print(resp.answer)
    print()


async def _one_shot(args) -> None:
    config = LegalAIConfig()
    engine = InsightEngine(config)
    ctx_text = _read_context_file(args.context) if args.context else ""

    resp = await engine.ask(
        question=args.question,
        mode=_parse_mode(args.mode),
        extra_context=ctx_text,
        use_context=bool(ctx_text),
    )
    _print_response(resp)


async def _repl() -> None:
    config = LegalAIConfig()
    engine = InsightEngine(config)
    ctx_text = ""

    print()
    print("═" * 72)
    print("  Legal AI — Insight REPL")
    print("═" * 72)
    print("Type a question to ask in the current mode.")
    print("Commands:")
    print("  /mode <name>     — switch mode (evaluate, elaborate, critique,")
    print("                     brainstorm, explain, open)")
    print("  /modes           — list available modes")
    print("  /context <path>  — load a file as case context")
    print("  /clear           — clear loaded context")
    print("  /quit or /exit   — leave")
    print()

    mode = InsightMode.OPEN
    while True:
        try:
            line = input(f"[{mode.value}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in ("quit", "exit"):
                break
            if cmd == "modes":
                _print_modes()
                continue
            if cmd == "mode":
                if not rest.strip():
                    print(f"Current mode: {mode.value}")
                else:
                    mode = _parse_mode(rest.strip())
                    print(f"Mode set to: {mode.value}")
                continue
            if cmd == "context":
                if not rest.strip():
                    print("Usage: /context <file path>")
                    continue
                ctx_text = _read_context_file(rest.strip())
                print(f"Loaded {len(ctx_text)} chars of context.")
                continue
            if cmd == "clear":
                ctx_text = ""
                print("Context cleared.")
                continue
            print(f"Unknown command: /{cmd}")
            continue

        resp = await engine.ask(
            question=line,
            mode=mode,
            extra_context=ctx_text,
            use_context=bool(ctx_text),
        )
        _print_response(resp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the legal AI for insights in different modes.",
    )
    parser.add_argument("question", nargs="?", help="Question to ask (omit for REPL)")
    parser.add_argument("--mode", default="open",
                        help="Insight mode (default: open)")
    parser.add_argument("--context", default=None,
                        help="Path to a file with case context")
    parser.add_argument("--list-modes", action="store_true",
                        help="Show available modes and exit")
    args = parser.parse_args()

    if args.list_modes:
        _print_modes()
        return

    if args.question:
        asyncio.run(_one_shot(args))
    else:
        asyncio.run(_repl())


if __name__ == "__main__":
    main()
