"""
Insight Engine — answers questions, evaluates ideas, and elaborates on
strategy in six distinct modes.  Designed to be invoked from both the GUI
(Insights tab) and the CLI (scripts/insights.py).

Each mode shapes the prompt so the LLM produces a response tailored to the
kind of help requested — Evaluate yields pros/cons + a recommendation,
Critique assumes the idea is already chosen and finds risks, Brainstorm
generates alternatives, and so on.

Optionally injects the user's current case context (situation, evidence
summary, generated documents) so answers are grounded in the active matter
rather than abstract advice.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from configs.config import LegalAIConfig


# ═══════════════════════════════════════════════════════════════════════════════
# MODES
# ═══════════════════════════════════════════════════════════════════════════════

class InsightMode(str, Enum):
    EVALUATE   = "evaluate"     # Pros/cons + recommendation
    ELABORATE  = "elaborate"    # Expand with detail
    CRITIQUE   = "critique"     # Find risks/blind spots
    BRAINSTORM = "brainstorm"   # Generate alternatives
    EXPLAIN    = "explain"      # Break down a concept/statute/case
    OPEN       = "open"         # Free-form Q&A


_MODE_INSTRUCTIONS: Dict[InsightMode, str] = {
    InsightMode.EVALUATE: """\
The user is asking whether the idea, plan, or decision below is a good idea.
Give a clear, candid evaluation in this structure:

  RECOMMENDATION (one sentence — yes / no / it depends, and the single
                  most important reason):
  <recommendation>

  STRONGEST REASONS IN FAVOR (2-4 bullets):
  • <reason>
  • <reason>

  STRONGEST REASONS AGAINST (2-4 bullets):
  • <reason>
  • <reason>

  CONDITIONS THAT WOULD CHANGE THE ANSWER (1-3 bullets):
  • <condition>

  BOTTOM LINE (one sentence the user can act on):
  <bottom line>

Be honest. Don't hedge to the point of uselessness — pick a side and defend it.""",

    InsightMode.ELABORATE: """\
The user wants more detail on the idea, concept, or plan below.
Expand it into a concrete, fully-developed treatment:

  THE CORE OF IT (2-3 sentences — what the idea actually means in practice):
  <core>

  DETAILED BREAKDOWN (5-8 specific points or sub-components):
  1. <point with substance — not a one-liner>
  2. <point>
  ...

  CONCRETE EXAMPLES (2-3 worked examples or illustrations):
  • <example>

  NEXT STEPS (3-5 specific actions to move this forward):
  • <action>

Be substantive. No vague platitudes. Each point should add real information.""",

    InsightMode.CRITIQUE: """\
The user is going to do the thing below — your job is to identify the risks,
blind spots, and likely failure modes BEFORE they happen.

  TOP RISKS (3-5 specific risks, ordered by severity):
  1. <risk — what specifically could go wrong and how it would play out>
  2. <risk>
  ...

  BLIND SPOTS (things the user probably hasn't considered):
  • <blind spot>
  • <blind spot>

  COMMON FAILURE MODES (how people typically get this wrong):
  • <failure mode>

  MITIGATIONS (specific things to do to reduce each top risk):
  1. <mitigation for risk #1>
  2. <mitigation for risk #2>
  ...

  IF YOU MUST PROCEED, the single most important thing to get right is:
  <one sentence>

Be direct. The user wants to know what could bite them — don't sugar-coat.""",

    InsightMode.BRAINSTORM: """\
The user wants alternative approaches to the question, problem, or decision
below.  Generate a diverse set of options — don't converge on one answer.

  OPTION 1 — <short title>:
    Approach: <one-paragraph description>
    Best when: <conditions where this wins>
    Trade-off: <what it costs>

  OPTION 2 — <short title>:
    [same format]

  OPTION 3 — <short title>:
    [same format]

  [2-4 more if genuinely distinct options exist]

  UNCONVENTIONAL OPTION (something the user almost certainly hasn't thought of):
    <description and why it might work>

  WHICH TO PICK FIRST IF YOU HAD TO CHOOSE BLIND:
    <option name and one-sentence reason>

Give genuinely different options — don't list five variations of the same idea.""",

    InsightMode.EXPLAIN: """\
The user wants a clear, thorough explanation of the concept, statute, case,
doctrine, or document section below.

  IN ONE SENTENCE: <plain-English summary>

  WHAT IT MEANS IN PRACTICE (3-5 sentences):
  <practical meaning>

  KEY ELEMENTS / REQUIREMENTS:
  1. <element with brief explanation>
  2. <element>
  ...

  COMMON MISCONCEPTIONS:
  • <misconception and the correction>

  HOW IT APPLIES TO A TYPICAL CASE:
  <one paragraph walking through application to a representative scenario>

  RELATED CONCEPTS / STATUTES / CASES TO KNOW:
  • <related thing and why it matters>

Be precise — when citing a statute or case, use exact section numbers and citations.""",

    InsightMode.OPEN: """\
Answer the user's question directly and substantively.  Structure your
response however best fits the question.  Be specific, cite authority where
relevant, and don't pad with disclaimers — get to the substance.""",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InsightResponse:
    """The result of a single insight query."""
    mode:           InsightMode
    question:       str
    answer:         str
    used_context:   bool                = False
    context_summary: str                = ""    # one-liner describing what was injected
    error:          Optional[str]       = None


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class InsightEngine:
    """
    Mode-driven Q&A engine for ad-hoc insights about ideas, plans, or
    questions.  When given case context, grounds answers in the active matter.
    """

    def __init__(self, config: LegalAIConfig):
        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed.")
        if not config.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set.")
        self.config = config
        genai.configure(api_key=config.google_api_key)

    async def ask(
        self,
        question:        str,
        mode:            InsightMode             = InsightMode.OPEN,
        case_situation:  str                     = "",
        evidence_summary: str                    = "",
        generated_docs:  Optional[List[str]]     = None,   # raw doc text snippets
        strategy_plan:   str                     = "",
        extra_context:   str                     = "",
        use_context:     bool                    = True,
    ) -> InsightResponse:
        """
        Ask a question in the chosen mode.

        Returns an InsightResponse — never raises; LLM failures are captured
        in response.error.
        """
        question = (question or "").strip()
        if not question:
            return InsightResponse(
                mode=mode, question="", answer="",
                error="Empty question — nothing to ask.",
            )

        context_block, summary = self._build_context_block(
            use_context=use_context,
            case_situation=case_situation,
            evidence_summary=evidence_summary,
            generated_docs=generated_docs or [],
            strategy_plan=strategy_plan,
            extra_context=extra_context,
        )

        mode_instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS[InsightMode.OPEN])

        prompt = self._build_prompt(question, mode_instruction, context_block)

        try:
            model = genai.GenerativeModel(self.config.model_pro)
            gen_cfg = GenerationConfig(temperature=0.4, max_output_tokens=8192)
            response = await asyncio.to_thread(
                model.generate_content, prompt, generation_config=gen_cfg,
            )
            answer = (response.text or "").strip()
        except Exception as exc:
            return InsightResponse(
                mode=mode, question=question, answer="",
                used_context=bool(context_block),
                context_summary=summary,
                error=f"LLM call failed: {exc}",
            )

        if not answer:
            return InsightResponse(
                mode=mode, question=question, answer="",
                used_context=bool(context_block),
                context_summary=summary,
                error="LLM returned empty response.",
            )

        return InsightResponse(
            mode=mode,
            question=question,
            answer=answer,
            used_context=bool(context_block),
            context_summary=summary,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_context_block(
        self,
        use_context:      bool,
        case_situation:   str,
        evidence_summary: str,
        generated_docs:   List[str],
        strategy_plan:    str,
        extra_context:    str,
    ) -> tuple[str, str]:
        """Return (context block string, short summary of what was included)."""
        if not use_context:
            return "", "No case context used — general-purpose answer."

        parts: List[str] = []
        included: List[str] = []

        if case_situation.strip():
            parts.append(f"CASE SITUATION:\n{case_situation.strip()}")
            included.append("situation")

        if evidence_summary.strip():
            parts.append(f"EVIDENCE SUMMARY:\n{evidence_summary.strip()}")
            included.append("evidence")

        if strategy_plan.strip():
            parts.append(f"STRATEGY CONTEXT:\n{strategy_plan.strip()}")
            included.append("strategy")

        if generated_docs:
            joined = "\n\n---\n\n".join(d.strip() for d in generated_docs if d.strip())
            if joined:
                parts.append(f"GENERATED DOCUMENTS (excerpts):\n{joined}")
                included.append(f"{len(generated_docs)} doc(s)")

        if extra_context.strip():
            parts.append(f"ADDITIONAL CONTEXT:\n{extra_context.strip()}")
            included.append("extra")

        if not parts:
            return "", "No case context available — general-purpose answer."

        block = (
            "═══ ACTIVE CASE CONTEXT ═══\n"
            "Use this as the factual backdrop when answering. If the question is\n"
            "about something else entirely, you may answer generally instead.\n\n"
            + "\n\n".join(parts)
            + "\n═══════════════════════════"
        )
        return block, "Used: " + ", ".join(included)

    def _build_prompt(
        self, question: str, mode_instruction: str, context_block: str,
    ) -> str:
        ctx_section = f"\n{context_block}\n" if context_block else ""
        return f"""You are a senior legal strategist providing direct, useful
insights to an attorney or pro se litigant.

{mode_instruction}
{ctx_section}
USER QUESTION:
{question}

Answer now, following the structure specified above:"""
