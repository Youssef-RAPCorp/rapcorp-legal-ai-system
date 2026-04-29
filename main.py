#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         RAPCorp LEGAL AI SYSTEM                               ║
║              Renaissance of American Physics and Astronomy (RAPCorp)          ║
║                                                                               ║
║  A comprehensive Legal AI platform powered by:                                ║
║  • Gemini Models:                                                             ║
║    - gemini-flash-lite-latest  (ultra-fast, cheapest)                         ║
║    - gemini-flash-latest       (balanced speed/quality)                       ║
║    - gemini-pro-latest         (maximum reasoning)                            ║
║                                                                               ║
║  • Customizable State Law Infrastructure                                      ║
║  • 15-Agent Knowledge Swarm                                                   ║
║  • IRAC/CREAC Reasoning Engine                                                ║
║  • 3-Stage RAG Pipeline                                                       ║
║  • Evidence Analyzer (audio, video, text → relevant evidence)                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python main.py                              # Run interactive demo
    python main.py --research "..."             # Quick research query
    python main.py --state CA                   # Set jurisdiction
    python main.py --add-state                  # Add new state config
    python main.py --evidence \\                 # Analyze evidence (inline situation)
        --situation "Case description..." \\
        --files recording.mp3 contract.pdf notes.txt
    python main.py --evidence \\                 # Analyze evidence (situation from file)
        --situation-file case_situation.txt \\
        --files recording.mp3 contract.pdf notes.txt
"""

import asyncio
import argparse
import sys
import os
import re
import json
import webbrowser
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import create_config, USState, LegalDomain
from src.legal_ai_system import LegalAIOrchestrator, create_legal_ai_system


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _md_to_html(text: str) -> str:
    """Convert common markdown patterns to HTML for research export."""
    # Escape HTML entities first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Code blocks (``` ... ```)
    text = re.sub(r'```[^\n]*\n(.*?)```', lambda m: f'<pre><code>{m.group(1)}</code></pre>',
                  text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Headings
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',   r'<h1>\1</h1>', text, flags=re.MULTILINE)
    # Bold / italic
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__',     r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_',       r'<em>\1</em>', text)
    # Horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '<hr>', text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r'^&gt;\s?(.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    # Unordered list items (group them)
    text = re.sub(r'^[-*•] (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*?</li>\n?)+', lambda m: f'<ul>{m.group(0)}</ul>', text, flags=re.DOTALL)
    # Ordered list items
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    # Paragraphs — blank lines become <p> breaks
    parts = re.split(r'\n{2,}', text)
    parts = [p.strip() for p in parts if p.strip()]
    result = []
    for part in parts:
        if part.startswith(('<h', '<ul', '<ol', '<pre', '<hr', '<blockquote')):
            result.append(part)
        else:
            result.append(f'<p>{part.replace(chr(10), "<br>")}</p>')
    return "\n".join(result)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Legal Research — {query_escaped}</title>
<style>
  body {{
    font-family: Georgia, "Times New Roman", serif;
    max-width: 900px; margin: 0 auto; padding: 0 24px 48px;
    color: #1a1a1a; line-height: 1.75; background: #fafafa;
  }}
  .banner {{
    background: #1a3a5c; color: white;
    padding: 20px 24px; margin: 0 -24px 32px;
  }}
  .banner h1 {{ margin: 0; font-size: 1.1em; font-family: sans-serif; font-weight: 700; }}
  .banner .query {{ margin: 6px 0 0; color: #adc8e8; font-size: 0.95em; font-family: sans-serif; }}
  h1 {{ font-size: 1.5em; color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; }}
  h2 {{ font-size: 1.2em; color: #1a3a5c; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 28px; }}
  h3 {{ font-size: 1.05em; color: #2a4a6c; margin-top: 20px; }}
  strong {{ font-weight: bold; }}
  em {{ font-style: italic; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 0.88em; }}
  pre {{ background: #f0f0f0; padding: 14px 18px; border-radius: 5px; overflow-x: auto; border-left: 3px solid #1a3a5c; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 4px solid #1a3a5c; margin: 16px 0; padding: 4px 16px; color: #444; background: #f4f8fc; }}
  ul, ol {{ padding-left: 26px; }}
  li {{ margin: 5px 0; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 24px 0; }}
  p {{ margin: 10px 0; }}
  .meta {{
    margin-top: 40px; padding-top: 14px; border-top: 1px solid #ddd;
    font-size: 0.82em; font-family: monospace; color: #777;
  }}
</style>
</head>
<body>
<div class="banner">
  <h1>RAPCorp Legal AI — Research Report</h1>
  <div class="query">{query_escaped}</div>
</div>
{body}
<div class="meta">
  Jurisdiction: {state} &nbsp;|&nbsp; Domain: {domain} &nbsp;|&nbsp; Mode: {mode}<br>
  Duration: {duration:.2f}s &nbsp;|&nbsp; Est. Cost: ${cost:.6f}<br>
  Generated: {timestamp}
</div>
</body>
</html>
"""


def _export_research(result: dict, query: str, state: str, domain: str,
                     mode: str, fmt: str) -> None:
    """Save research result in the requested format(s) and open HTML in browser."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = PROJECT_ROOT / "output" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r'[^a-z0-9]+', '_', query.lower())[:40].strip('_')
    base = out_dir / f"{timestamp}_{slug}"

    response_text = result.get("response", "")
    do_html = fmt in ("html", "all")
    do_md   = fmt in ("md",   "all")
    do_json = fmt in ("json", "all")

    if do_md or do_html:
        md_path = base.with_suffix(".md")
        md_content = (
            f"# Legal Research: {query}\n\n"
            f"**Jurisdiction:** {state} | **Domain:** {domain} | **Mode:** {mode}  \n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
            f"**Duration:** {result.get('duration_seconds', 0):.2f}s | "
            f"**Cost:** ${result.get('total_cost', 0):.6f}\n\n---\n\n"
            + response_text
        )
        md_path.write_text(md_content, encoding="utf-8")
        if do_md:
            print(f"  Markdown saved → {md_path}")

    if do_html:
        html_path = base.with_suffix(".html")
        body_html = _md_to_html(response_text)
        html = _HTML_TEMPLATE.format(
            query_escaped=query.replace('"', "&quot;"),
            body=body_html,
            state=state, domain=domain, mode=mode,
            duration=result.get("duration_seconds", 0),
            cost=result.get("total_cost", 0),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        html_path.write_text(html, encoding="utf-8")
        print(f"  HTML report  → {html_path}")
        webbrowser.open(html_path.as_uri())

    if do_json:
        json_path = base.with_suffix(".json")
        payload = {
            "query": query, "state": state, "domain": domain, "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": result.get("duration_seconds"),
            "total_cost": result.get("total_cost"),
            "response": response_text,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON export  → {json_path}")


async def run_research(
    query: str,
    state: str = "FED",
    domain: str = "contract",
    mode: str = "standard",
    export: str = "html",
):
    """Run a quick research query and export the result."""
    print(f"\n🔬 Researching: {query}")
    print(f"   Jurisdiction: {state} | Domain: {domain} | Mode: {mode}")
    print("-" * 60)

    system = await create_legal_ai_system()

    try:
        state_enum = USState(state.upper())
    except ValueError:
        print(f"⚠️ Unknown state: {state}, using Federal")
        state_enum = USState.FEDERAL

    try:
        domain_enum = LegalDomain(domain.lower())
    except ValueError:
        print(f"⚠️ Unknown domain: {domain}, using Contract")
        domain_enum = LegalDomain.CONTRACT

    result = await system.research(
        query=query,
        state=state_enum,
        domain=domain_enum,
        mode=mode
    )

    print("\n📋 RESULT:")
    print("=" * 60)
    print(result["response"])
    print("\n" + "=" * 60)
    print(f"Duration: {result['duration_seconds']:.2f}s | Cost: ${result['total_cost']:.6f}")

    if export != "none":
        print("\n📄 Exporting…")
        _export_research(result, query, state, domain, mode, export)


async def interactive_mode():
    """Run the interactive demo."""
    from src.legal_ai_system import demo
    await demo()


def _print_separator(title: str = "") -> None:
    line = "=" * 70
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(f"\n{line}")


def _ask(prompt: str) -> str:
    """Print a prompt and return the stripped user input."""
    print(f"\n{prompt}")
    return input("  > ").strip()


async def run_case_chat(
    directory: str,
    state: str = "FED",
    situation_update: Optional[str] = None,
) -> None:
    """
    Interactive chat with the AI Lawyer about all documents in a case directory.

    The AI loads every document once at the start, then answers questions
    conversationally with full knowledge of the case file. Conversation
    history is preserved in-session so follow-up questions work naturally.

    Special commands (type at the prompt):
        summary    — Case overview and current status
        timeline   — Chronological event list
        risks      — Outstanding issues and urgency items
        recommend  — Next-step recommendations
        docs       — List documents loaded
        quit / exit — End the session
    """
    _print_separator("RAPCORP LEGAL AI — CASE CHAT")
    print(f"  Directory    : {directory}")
    print(f"  Jurisdiction : {state}")

    # ── Scan documents ─────────────────────────────────────────────────────
    from src.documents.case_reader import CaseDirectoryScanner
    scanner = CaseDirectoryScanner()

    try:
        documents = scanner.scan(directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"\n  Error: {e}")
        return

    stats = scanner.get_stats(documents)
    readable = [d for d in documents if not d.read_error and d.char_count > 0]

    if not readable:
        print("\n  No readable documents found in that directory.")
        return

    print(f"\n  Loaded {stats['readable']} document(s) into context:")
    for doc in readable:
        print(f"    [{doc.doc_type:15s}] {doc.filename}  ({doc.char_count:,} chars)")

    context_block = scanner.build_context_block(documents)

    # ── Initialize system ──────────────────────────────────────────────────
    system = await create_legal_ai_system()

    try:
        state_enum = USState(state.upper())
    except ValueError:
        print(f"  Unknown state '{state}', using Federal.")
        state_enum = USState.FEDERAL

    if not system.llm:
        print("\n  Error: LLM not initialized. Check your GOOGLE_API_KEY in .env.")
        return

    # ── Build the persistent system prompt ────────────────────────────────
    update_section = (
        f"\n\nNEW SITUATION UPDATE FROM CLIENT (most recent context — prioritize this):\n"
        f"{situation_update}\n--- END OF UPDATE ---"
    ) if situation_update else ""

    system_prompt = f"""You are an experienced litigation attorney and legal advisor \
reviewing a client's complete case file. You have deep knowledge of all documents \
listed below and you answer questions about this specific case with precision and care.

JURISDICTION: {state_enum.value}{update_section}

FULL CASE FILE ({stats['readable']} documents):
{context_block}

Guidelines:
- Always answer in the context of THIS specific case using the actual documents above
- Reference specific document names, dates, and verbatim excerpts when relevant
- Flag any deadlines, risks, or urgency items proactively
- Be direct and practical — the client is acting pro se
- Remind the client to consult a licensed attorney before taking legal action
- If something is not clear from the documents, say so rather than speculating"""

    # Conversation history: list of {"role": "user"/"model", "parts": [{"text": "..."}]}
    history: List[Dict] = []

    # ── Quick-start shortcuts ─────────────────────────────────────────────
    shortcuts = {
        "summary":   "Give me a concise summary of this case: the parties, the core claims, and the current status.",
        "timeline":  "List all key events and filings in this case in chronological order.",
        "risks":     "What are the outstanding legal issues, risks, and any time-sensitive deadlines in this case?",
        "recommend": "What are your top recommendations for my next steps in this case?",
        "docs":      None,  # handled locally
    }

    print(f"\n  Chat is ready. Type your question, a shortcut, or 'quit' to exit.")
    print(f"  Shortcuts: summary | timeline | risks | recommend | docs")
    print(f"  {'─'*60}")

    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig

    # Use a stateful chat session so history is preserved automatically
    model = genai.GenerativeModel(
        system.config.model_pro,
        system_instruction=system_prompt,
    )
    chat_session = model.start_chat(history=[])

    gen_config = GenerationConfig(
        temperature=0.4,
        max_output_tokens=8192,
    )

    while True:
        try:
            raw = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Chat ended.")
            break

        if not raw:
            continue

        lower = raw.lower()

        # Exit commands
        if lower in ("quit", "exit", "q", "bye"):
            print("\n  Chat ended. Goodbye.")
            break

        # Local shortcut: list documents
        if lower == "docs":
            print(f"\n  Documents loaded ({stats['readable']}):")
            for doc in readable:
                print(f"    [{doc.doc_type:15s}]  {doc.filename}")
            continue

        # Map shortcut to full question
        question = shortcuts.get(lower, raw)

        # Send to AI
        print("\n  AI Lawyer: ", end="", flush=True)
        try:
            import asyncio as _asyncio
            response = await _asyncio.to_thread(
                chat_session.send_message,
                question,
                generation_config=gen_config,
            )
            answer = response.text if hasattr(response, "text") else "(No response)"
        except Exception as e:
            answer = f"(Error communicating with AI: {e})"

        # Print with light indentation
        print()
        for line in answer.splitlines():
            print(f"  {line}")

        print(f"\n  {'─'*60}")


async def run_case_directory_analysis(
    directory: str,
    state: str = "FED",
    situation_update: Optional[str] = None,
) -> None:
    """
    Load all documents from a case directory, analyze the full case
    history, display recommendations, and optionally draft follow-up
    documents.

    Phase 1: Scan directory — show what was found
    Phase 2: AI case analysis — overview, timeline, status, recommendations
    Phase 3: Optional follow-up document generation
    """
    from src.legal_ai_system import CaseDirectoryAnalysis

    _print_separator("RAPCORP LEGAL AI — CASE DIRECTORY ANALYSIS")
    print(f"  Directory    : {directory}")
    print(f"  Jurisdiction : {state}")
    if situation_update:
        preview = situation_update[:120].replace("\n", " ")
        print(f"  Update       : \"{preview}{'...' if len(situation_update) > 120 else ''}\"")


    # ── Quick scan preview (no AI yet) ────────────────────────────────────
    from src.documents.case_reader import CaseDirectoryScanner
    scanner = CaseDirectoryScanner()
    try:
        documents = scanner.scan(directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"\n  Error: {e}")
        return

    stats = scanner.get_stats(documents)
    print(f"\n  Documents found   : {stats['total']}")
    print(f"  Readable          : {stats['readable']}")
    if stats["errors"]:
        print(f"  Could not read    : {stats['errors']} ({', '.join(stats['error_files'])})")
    if stats["by_type"]:
        print(f"  Types detected    :", end="")
        type_parts = [f"{count}x {dtype}" for dtype, count in stats["by_type"].items()]
        print("  " + ", ".join(type_parts))

    if stats["readable"] == 0:
        print("\n  No readable documents found. Check the directory path and file formats.")
        return

    print("\n  Documents to analyze:")
    for doc in documents:
        status = f"[{doc.doc_type}]" if not doc.read_error else "[ERROR]"
        chars = f"  {doc.char_count:,} chars" if doc.char_count else ""
        print(f"    {status:20s}  {doc.filename}{chars}")

    confirm = _ask("\nProceed with AI analysis of all documents? (y/n, default y)").lower()
    if confirm == "n":
        print("  Cancelled.")
        return

    # ── Initialize system ─────────────────────────────────────────────────
    system = await create_legal_ai_system()

    try:
        state_enum = USState(state.upper())
    except ValueError:
        print(f"  Unknown state '{state}', using Federal.")
        state_enum = USState.FEDERAL

    # ── Phase 2: AI analysis ──────────────────────────────────────────────
    _print_separator("PHASE 1 — CASE ANALYSIS")
    print("  Analyzing case file... (this may take a moment)")

    analysis: CaseDirectoryAnalysis = await system.analyze_case_directory(
        directory=directory,
        state=state_enum,
        situation_update=situation_update,
    )

    # ── Display results ───────────────────────────────────────────────────
    print(f"\n  Documents read : {analysis.documents_read} / {analysis.documents_found}")
    print(f"  Model used     : {analysis.model_used}")
    print(f"  Est. cost      : ${analysis.cost:.6f}")

    print(f"\n{'='*65}")
    print("  CASE OVERVIEW")
    print(f"{'='*65}")
    print(f"\n  {analysis.case_overview}")

    if analysis.parties:
        print("\n  PARTIES:")
        for role, name in analysis.parties.items():
            print(f"    {role.replace('_', ' ').title():<30} {name}")

    print(f"\n  Your role: {analysis.user_role.upper()}")

    if analysis.timeline:
        print(f"\n{'='*65}")
        print("  CASE TIMELINE")
        print(f"{'='*65}")
        for event in analysis.timeline:
            print(f"    • {event}")

    print(f"\n{'='*65}")
    print("  CURRENT STATUS")
    print(f"{'='*65}")
    print(f"\n  {analysis.current_status}")

    if analysis.urgency_items:
        print(f"\n{'='*65}")
        print("  TIME-SENSITIVE ITEMS  *** ATTENTION REQUIRED ***")
        print(f"{'='*65}")
        for item in analysis.urgency_items:
            print(f"    ⚠  {item}")

    if analysis.outstanding_issues:
        print(f"\n{'='*65}")
        print("  OUTSTANDING ISSUES")
        print(f"{'='*65}")
        for i, issue in enumerate(analysis.outstanding_issues, 1):
            print(f"    {i}. {issue}")

    print(f"\n{'='*65}")
    print("  RECOMMENDATIONS")
    print(f"{'='*65}")
    for i, rec in enumerate(analysis.recommendations, 1):
        print(f"    {i}. {rec}")

    if analysis.suggested_next_documents:
        print(f"\n{'='*65}")
        print("  IF YOU NEED TO FILE — SUGGESTED DOCUMENTS")
        print(f"{'='*65}")
        for i, doc in enumerate(analysis.suggested_next_documents, 1):
            print(f"\n    [{i}] {doc.get('title', doc.get('type', 'Document'))}")
            print(f"         Type   : {doc.get('type', 'unknown')}")
            print(f"         Reason : {doc.get('reason', '')}")

    # ── Counsel complete — document generation is optional ────────────────
    print(f"\n{'='*65}")
    print("  COUNSEL COMPLETE")
    print(f"{'='*65}")
    print("\n  The analysis above is your legal guidance for this case.")
    print("  Use it to prepare for court, understand your position,")
    print("  and decide on your next steps.")
    print("\n  REMINDER: Consult a licensed attorney before filing or appearing in court.")

    # ── Optional: draft a document only if user explicitly asks ──────────
    if not analysis.suggested_next_documents:
        return

    gen = _ask(
        "\n  Would you also like to draft a follow-up document? (y/n, default n)"
    ).lower()
    if gen != "y":
        print("\n  No document drafted. Your counsel summary is above.")
        return

    _print_separator("OPTIONAL — DRAFT A DOCUMENT")
    print("\n  Available document types:")
    for i, doc in enumerate(analysis.suggested_next_documents, 1):
        print(f"    ({i}) {doc.get('title', doc.get('type'))}")
    print(f"    (0) Cancel")

    raw = _ask("Your choice (number)").strip()
    try:
        choice_idx = int(raw)
    except ValueError:
        choice_idx = 0

    if choice_idx == 0 or choice_idx > len(analysis.suggested_next_documents):
        print("\n  Cancelled.")
        return

    selected = analysis.suggested_next_documents[choice_idx - 1]
    doc_type = selected.get("type", "motion")

    instructions = _ask(
        f"Any specific instructions for the {doc_type.replace('_', ' ')} "
        f"(press ENTER for none)"
    ).strip()

    print(f"\n  Drafting {doc_type.replace('_', ' ')}...")
    result = await system.generate_case_continuation(
        analysis=analysis,
        document_type=doc_type,
        additional_instructions=instructions,
    )

    _print_separator("DOCUMENT DRAFTED")
    print(f"\n  Title       : {result['title']}")
    print(f"  Saved to    : {result['file_path']}")
    print(f"  Sign        : {'Yes' if result['requires_signature'] else 'No'}")
    print(f"  Notarize    : {'Yes' if result['requires_notarization'] else 'No'}")
    print(f"\n  Review this document carefully before using it in court.")


async def run_preflight_dialogue(
    system,
    situation: str,
    state_enum,
    files: List[str],
) -> tuple:
    """
    AI pre-flight check with optional clarification dialogue.

    - If the AI finds no issues, proceeds silently.
    - If issues are found, shows them and gives the user a choice:
        (1) Answer questions one at a time
        (2) Proceed without clarification
    - Returns (refined_situation, preflight_clarifications_dict).
    """
    from src.legal_ai_system import PreFlightResult

    _print_separator("PRE-FLIGHT CHECK")
    print("  Reviewing your case situation before proceeding...")

    preflight: PreFlightResult = await system.pre_flight_check(
        situation_description=situation,
        state=state_enum,
        files=files,
    )

    # Always show the AI's understanding — useful even when no issues are found
    print(f"\n  AI Understanding:\n  {preflight.situation_summary}")
    print(f"  Confidence: {preflight.overall_confidence:.0%}")

    if not preflight.has_issues:
        print("\n  No issues found. Proceeding with analysis.")
        return situation, {}

    # Categorize issues by severity
    critical = [i for i in preflight.issues if i.severity == "critical"]
    moderate = [i for i in preflight.issues if i.severity == "moderate"]
    minor    = [i for i in preflight.issues if i.severity == "minor"]

    print(f"\n  Found {len(preflight.issues)} item(s) that may need clarification:")
    if critical:
        print(f"    Critical : {len(critical)}")
    if moderate:
        print(f"    Moderate : {len(moderate)}")
    if minor:
        print(f"    Minor    : {len(minor)}")

    for idx, issue in enumerate(preflight.issues, 1):
        tag = issue.severity.upper()
        print(f"\n  {idx}. [{tag}] {issue.description}")

    # Ask user preference
    print("\n  How would you like to proceed?")
    print("  (1) Answer clarifying questions")
    print("  (2) Proceed without clarification")
    choice = _ask("Your choice (1 or 2, default 1)").strip() or "1"

    if choice != "1":
        print("\n  Proceeding without pre-flight clarification.")
        return situation, {}

    # Collect answers one at a time
    clarifications: dict = {}
    for idx, issue in enumerate(preflight.issues, 1):
        tag = issue.severity.upper()
        answer = _ask(f"[{idx}/{len(preflight.issues)}] [{tag}] {issue.question}")
        if answer:
            clarifications[issue.question] = answer

    if not clarifications:
        print("\n  No answers provided. Proceeding as-is.")
        return situation, {}

    # Fold clarifications into the situation description so every downstream
    # component (evidence analyzer, document generator) sees the full context.
    clarification_block = "\n".join(
        f"- {q}: {a}" for q, a in clarifications.items()
    )
    refined_situation = (
        f"{situation}\n\n"
        f"ADDITIONAL CONTEXT (provided before analysis):\n{clarification_block}"
    )

    print(f"\n  Pre-flight complete. {len(clarifications)} clarification(s) added to case context.")
    return refined_situation, clarifications


async def _ai_edit_document(config, file_path: str, instruction: str) -> None:
    """Apply an AI-assisted targeted edit to a generated document, in-place."""
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
    except ImportError:
        print("  google-generativeai not available — edit skipped.")
        return

    content = Path(file_path).read_text(encoding="utf-8")
    genai.configure(api_key=config.google_api_key)
    model = genai.GenerativeModel(config.model_pro)

    prompt = f"""You are editing a legal document. Apply ONLY the change described below.
Do not add preamble, commentary, or notes — return the complete revised document only.
Preserve all formatting, structure, paragraph numbering, and content not mentioned in the change.

CHANGE REQUESTED:
{instruction}

DOCUMENT:
{content}

Write the complete revised document now:"""

    gen_config = GenerationConfig(temperature=0.1, max_output_tokens=16384)
    try:
        response = await asyncio.to_thread(
            model.generate_content, prompt, generation_config=gen_config
        )
        revised = (response.text or "").strip()
        if revised:
            Path(file_path).write_text(revised, encoding="utf-8")
        else:
            print("  AI returned empty output — file unchanged.")
    except Exception as exc:
        print(f"  AI edit failed: {exc}")


async def run_evidence_analysis(
    situation: str,
    files: List[str],
    state: str = "FED",
):
    """
    Interactive evidence analysis pipeline:
      1. Initial analysis of files
      2. Clarifying Q&A loop (resolves uncertainties without re-uploading)
      3. Optional document generation
    """
    _print_separator("RAPCORP LEGAL AI — EVIDENCE ANALYSIS")
    print(f"  Situation file : {len(situation)} characters loaded")
    print(f"  Evidence files : {', '.join(files)}")
    print(f"  Jurisdiction   : {state}")

    system = await create_legal_ai_system()

    try:
        state_enum = USState(state.upper())
    except ValueError:
        print(f"⚠️  Unknown state '{state}', using Federal.")
        state_enum = USState.FEDERAL

    # ── PRE-FLIGHT CHECK ──────────────────────────────────────────────────
    situation, preflight_clarifications = await run_preflight_dialogue(
        system, situation, state_enum, files
    )

    # ── PHASE 1: Initial analysis ─────────────────────────────────────────
    _print_separator("PHASE 1 — INITIAL EVIDENCE ANALYSIS")
    result = await system.analyze_evidence(
        situation_description=situation,
        file_paths=files,
        state=state_enum,
    )
    analysis = result["_result_obj"]
    analysis.print_report()
    print(f"\nEst. cost so far: ${result['cost']:.6f}")

    # ── PHASE 2: Interactive clarification loop ───────────────────────────
    all_clarifications: dict = {}
    round_num = 0

    while True:
        _print_separator("PHASE 2 — CLARIFICATION")
        print("  Identifying uncertainties in the analysis...")
        questions = await system.generate_clarifying_questions(
            analysis_result=analysis,
            situation_description=situation,
        )

        if not questions:
            print("\n  No significant uncertainties found. Analysis is complete.")
            break

        print(f"\n  The system has {len(questions)} clarifying question(s).")
        print("  Answer them to refine the analysis.")
        print("  Press ENTER to skip any question.")

        round_answers: dict = {}
        for i, q in enumerate(questions, 1):
            answer = _ask(f"[{i}/{len(questions)}] {q}")
            if answer:
                round_answers[q] = answer

        if not round_answers:
            print("\n  No answers provided — skipping refinement.")
            break

        all_clarifications.update(round_answers)

        print(f"\n  Refining analysis with {len(round_answers)} new answer(s)...")
        refined = await system.refine_evidence_analysis(
            original_result=analysis,
            situation_description=situation,
            clarifications=all_clarifications,
            state=state_enum,
        )
        analysis = refined["_result_obj"]

        _print_separator(f"REFINED ANALYSIS (Round {round_num + 1})")
        analysis.print_report()
        print(f"\n  Est. cost so far: ${analysis.cost:.6f}")
        round_num += 1

        cont = _ask("Continue clarifying? (y/n)").lower()
        if cont != "y":
            break

    # ── PHASE 3: Document generation ─────────────────────────────────────
    _print_separator("PHASE 3 — PETITION DOCUMENTS")
    gen = _ask("Generate all petition documents needed for filing? (y/n)").lower()
    if gen != "y":
        print("\n  Skipping document generation. Analysis saved above.")
        return

    print("\n  Generating petition package for " + state_enum.value + "...")
    merged_clarifications = {**preflight_clarifications, **all_clarifications}
    docs = await system.generate_petition_documents(
        situation=situation,
        analysis=analysis,
        clarifications=merged_clarifications,
        state=state_enum,
    )

    _print_separator("GENERATED FILES")
    for doc in docs:
        sig   = " [SIGN]"    if doc["requires_signature"]    else ""
        notary = " [NOTARIZE]" if doc["requires_notarization"] else ""
        filed  = ""           if doc["filing_required"]        else " [keep for records]"
        print(f"\n  {doc['filename']}{sig}{notary}{filed}")
        print(f"    {doc['title']}")
        print(f"    {doc['description']}")
        print(f"    Path: {doc['file_path']}")

    checklist = next((d for d in docs if d["doc_type"] == "checklist"), None)
    if checklist:
        print(f"\n  *** READ THIS FIRST: {checklist['file_path']} ***")

    total_to_file = sum(1 for d in docs if d["filing_required"])
    print(f"\n  Total documents generated : {len(docs)}")
    print(f"  Documents to file with court: {total_to_file}")
    print(f"\n  IMPORTANT: Review every document with a licensed attorney before filing.")

    # ── PHASE 4: Manual review & edit ────────────────────────────────────
    _print_separator("PHASE 4 — REVIEW & EDIT DOCUMENTS")
    print("  All documents are ready. You can open and edit them now.")
    print()
    print("  Commands:")
    print("    <number>      — open the document in your default text editor")
    print("    <number> ai   — describe a change; AI applies it and saves the file")
    print("    done          — finish")

    # Filter out review logs for the edit menu (they're auditing artefacts)
    editable = [d for d in docs if "REVIEW_LOG" not in d["filename"]]

    while True:
        print()
        for i, doc in enumerate(editable, 1):
            sig    = " [SIGN]"     if doc.get("requires_signature")    else ""
            notary = " [NOTARIZE]" if doc.get("requires_notarization") else ""
            print(f"  [{i:2d}] {doc['title']}{sig}{notary}")
            print(f"        {doc['file_path']}")
        print()

        raw = _ask("Choice (number / 'number ai' / 'done')").strip().lower()
        if not raw or raw in ("done", "exit", "quit"):
            break

        parts = raw.split()
        ai_mode = len(parts) == 2 and parts[1] == "ai"
        try:
            chosen = editable[int(parts[0]) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice — enter a number from the list above.")
            continue

        if ai_mode:
            instruction = _ask("Describe the change you want (be specific)")
            if not instruction:
                continue
            print("  Applying AI edit...")
            await _ai_edit_document(system.config, chosen["file_path"], instruction)
            print(f"  Saved: {chosen['file_path']}")
        else:
            path = chosen["file_path"]
            print(f"\n  Opening: {path}")
            if sys.platform == "win32":
                os.startfile(path)
            else:
                import subprocess
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, path])
            input("\n  Press ENTER when done editing...")


async def show_system_info():
    """Show system information."""
    print("\n" + "=" * 60)
    print("       RAPCorp LEGAL AI SYSTEM - SYSTEM INFO")
    print("=" * 60)
    
    config = create_config()
    
    print("\n📊 GEMINI MODELS:")
    print(f"   Flash Lite: {config.model_flash_lite}")
    print(f"   Flash:      {config.model_flash}")
    print(f"   Pro:        {config.model_pro}")
    
    print("\n🗺️ ENABLED STATES:")
    for state in config.enabled_states:
        state_config = config.get_state_config(state)
        if state_config:
            status = "✅" if state_config.is_active else "⚠️"
            print(f"   {status} {state.value}: {state_config.state_name}")
    
    print("\n💰 COST MANAGEMENT:")
    print(f"   Daily Budget:   ${config.daily_budget_usd:.2f}")
    print(f"   Monthly Budget: ${config.monthly_budget_usd:.2f}")
    
    print("\n🔧 CONFIGURATION:")
    print(f"   Tracking Costs: {'Yes' if config.track_costs else 'No'}")
    print(f"   UPL Safeguards: {'Enabled' if config.upl_safeguards else 'Disabled'}")
    print(f"   Human Review:   {'Required' if config.require_human_review else 'Optional'}")
    
    api_key = os.getenv("GOOGLE_API_KEY", "")
    print("\n🔑 API STATUS:")
    print(f"   GOOGLE_API_KEY: {'✅ Set' if api_key else '❌ Not Set'}")


def main():
    parser = argparse.ArgumentParser(
        description="RAPCorp Legal AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                          # Interactive demo
  python main.py --research "breach of contract"         # Quick research
  python main.py --research "..." --state CA             # California law
  python main.py --info                                  # System info
  python main.py --add-state                             # Add state config

  # Evidence analysis — situation typed inline
  python main.py --evidence --state CA \\
      --situation "Employer wrongfully terminated employee after whistleblowing." \\
      --files meeting.mp3 termination_letter.pdf email_chain.txt

  # Evidence analysis — situation loaded from a text file (recommended)
  python main.py --evidence --state CA \\
      --situation-file case_situation.txt \\
      --files meeting.mp3 termination_letter.pdf email_chain.txt
        """
    )
    
    parser.add_argument("--research", "-r", metavar="QUERY",
                        help="Run a legal research query")
    parser.add_argument("--state", "-s", default="FED",
                        help="State jurisdiction (default: FED for Federal)")
    parser.add_argument("--domain", "-d", default="contract",
                        help="Legal domain (contract/tort/criminal/etc)")
    parser.add_argument("--mode", "-m", default="standard",
                        choices=["quick", "standard", "comprehensive"],
                        help="Research mode (default: standard)")
    parser.add_argument("--export", default="html",
                        choices=["html", "md", "json", "all", "none"],
                        help="Export research result: html (default, opens in browser), "
                             "md, json, all, or none")
    parser.add_argument("--info", "-i", action="store_true",
                        help="Show system information")
    parser.add_argument("--add-state", "-a", action="store_true",
                        help="Run state addition wizard")

    # Case directory analysis
    parser.add_argument("--case-dir", metavar="DIR",
                        help="Analyze all documents in a case directory for follow-up guidance")
    parser.add_argument("--update", metavar="TEXT",
                        help="New situation text to add on top of existing case documents (use with --case-dir)")
    parser.add_argument("--update-file", metavar="PATH",
                        help="Path to a .txt file with a situation update (use with --case-dir)")

    # Case chat
    parser.add_argument("--chat", action="store_true",
                        help="Chat with the AI about a case directory (use with --case-dir)")

    # Evidence analysis
    parser.add_argument("--evidence", "-e", action="store_true",
                        help="Analyze files for legally relevant evidence")
    parser.add_argument("--situation",
                        help="Case situation description as inline text")
    parser.add_argument("--situation-file", metavar="PATH",
                        help="Path to a .txt file containing the case situation description")
    parser.add_argument("--files", nargs="+", metavar="FILE",
                        help="Files to analyze (audio, video, or text)")
    
    args = parser.parse_args()
    
    # Show info
    if args.info:
        asyncio.run(show_system_info())
        return
    
    # Add state
    if args.add_state:
        from scripts.add_state import interactive_add_state, save_state_config
        config = interactive_add_state()
        save = input("\nSave this configuration? (y/n): ").strip().lower()
        if save in ("y", "yes"):
            path = save_state_config(config)
            print(f"✅ Saved to: {path}")
        return
    
    # Evidence analysis
    if args.evidence:
        # Resolve situation text: file takes priority over inline
        situation_text = None
        if args.situation_file:
            sit_path = Path(args.situation_file)
            if not sit_path.exists():
                print(f"Error: situation file not found: {args.situation_file}")
                sys.exit(1)
            situation_text = sit_path.read_text(encoding="utf-8").strip()
            if not situation_text:
                print(f"Error: situation file is empty: {args.situation_file}")
                sys.exit(1)
            print(f"Loaded situation from: {sit_path.name} ({len(situation_text)} chars)")
        elif args.situation:
            situation_text = args.situation.strip()

        if not situation_text:
            print("Error: provide --situation 'text' or --situation-file path.txt")
            print("  Example: python main.py --evidence "
                  "--situation-file case.txt --files rec.mp3 doc.pdf")
            sys.exit(1)
        if not args.files:
            print("Error: --files is required when using --evidence.")
            print("  Example: python main.py --evidence "
                  "--situation-file case.txt --files rec.mp3 doc.pdf")
            sys.exit(1)
        asyncio.run(run_evidence_analysis(
            situation=situation_text,
            files=args.files,
            state=args.state,
        ))
        return

    # Case chat
    if args.chat:
        if not args.case_dir:
            print("Error: --chat requires --case-dir <path>")
            print("  Example: python main.py --chat --case-dir output/2026-03-06_07-21-35 --state FL")
            sys.exit(1)
        # Resolve situation update (same logic as --case-dir)
        chat_update = None
        if args.update_file:
            update_path = Path(args.update_file)
            if not update_path.exists():
                print(f"Error: update file not found: {args.update_file}")
                sys.exit(1)
            chat_update = update_path.read_text(encoding="utf-8").strip()
        elif args.update:
            chat_update = args.update.strip()
        asyncio.run(run_case_chat(
            directory=args.case_dir,
            state=args.state,
            situation_update=chat_update,
        ))
        return

    # Case directory analysis
    if args.case_dir:
        # Resolve situation update: file takes priority over inline text
        situation_update = None
        if args.update_file:
            update_path = Path(args.update_file)
            if not update_path.exists():
                print(f"Error: update file not found: {args.update_file}")
                sys.exit(1)
            situation_update = update_path.read_text(encoding="utf-8").strip()
            if not situation_update:
                print(f"Error: update file is empty: {args.update_file}")
                sys.exit(1)
            print(f"Loaded update from: {update_path.name} ({len(situation_update)} chars)")
        elif args.update:
            situation_update = args.update.strip()

        asyncio.run(run_case_directory_analysis(
            directory=args.case_dir,
            state=args.state,
            situation_update=situation_update,
        ))
        return

    # Research query
    if args.research:
        asyncio.run(run_research(
            query=args.research,
            state=args.state,
            domain=args.domain,
            mode=args.mode,
            export=args.export,
        ))
        return

    # Default: interactive mode
    asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
