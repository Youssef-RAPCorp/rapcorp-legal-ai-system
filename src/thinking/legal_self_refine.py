"""
Legal Self-Refiner — Generate → Critique → Refine loop.

Ported from Research-Agent thinking/self_refine.py and adapted for
legal argument planning.  Runs up to MAX_CYCLES cycles, stopping early
when the weighted rubric score exceeds QUALITY_THRESHOLD.

Legal rubric (6 criteria, weighted):
  factual_support      (2.0) — every claim backed by specific evidence
  legal_authority      (2.0) — statutes and case law cited for every argument
  argument_strength    (1.5) — reasoning is logical, not just conclusory
  completeness         (1.5) — all elements of the cause of action addressed
  procedural_soundness (1.0) — correct court, correct procedure, correct timing
  counter_resistance   (2.0) — argument survives opposing counsel's best attacks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

MAX_CYCLES       = 3
QUALITY_THRESHOLD = 8.0   # out of 10.0


@dataclass
class RefineCycle:
    cycle: int
    strategy_text: str
    scores: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    critique_feedback: str = ""


@dataclass
class RefineResult:
    final_strategy: str
    final_score: float
    cycles_completed: int
    history: List[RefineCycle] = field(default_factory=list)
    passed_threshold: bool = False


_RUBRIC = [
    ("factual_support",      2.0, "Every factual claim is supported by a specific, named piece of evidence (exhibit, document, timestamp)."),
    ("legal_authority",      2.0, "Every legal argument cites a specific statute by section number or a named court case with citation."),
    ("argument_strength",    1.5, "The reasoning flows logically from facts to law to conclusion without leaps or conclusory statements."),
    ("completeness",         1.5, "Every required element of the legal claim or defense is explicitly addressed."),
    ("procedural_soundness", 1.0, "The argument respects proper procedure, venue, standing, and timing."),
    ("counter_resistance",   2.0, "The argument anticipates and pre-empts the most damaging opposing attacks identified in the attack surface analysis."),
]

_TOTAL_WEIGHT = sum(w for _, w, _ in _RUBRIC)


class LegalSelfRefiner:
    """
    Generate → Critique → Refine loop for legal argument strategy.
    Uses the existing GeminiClient; no new external dependencies.
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def refine(
        self,
        situation: str,
        evidence_summary: str,
        case_law_context: str,
        attack_surface_summary: str,
        state: str,
        doc_mode: str = "petition",
    ) -> RefineResult:
        """Run the Generate → Critique → Refine loop and return the best strategy."""

        history: List[RefineCycle] = []
        best_text  = ""
        best_score = 0.0

        current_critique = ""
        current_strategy = ""

        for cycle_num in range(1, MAX_CYCLES + 1):
            # ── Generate ─────────────────────────────────────────────────
            current_strategy = await self._generate(
                situation, evidence_summary, case_law_context,
                attack_surface_summary, state, doc_mode,
                previous_strategy=current_strategy,
                previous_critique=current_critique,
                cycle=cycle_num,
            )

            # ── Critique ─────────────────────────────────────────────────
            scores, feedback = await self._critique(current_strategy)

            weighted = sum(
                scores.get(name, 5.0) * weight
                for name, weight, _ in _RUBRIC
            ) / _TOTAL_WEIGHT

            rec = RefineCycle(
                cycle=cycle_num,
                strategy_text=current_strategy,
                scores=scores,
                weighted_score=weighted,
                critique_feedback=feedback,
            )
            history.append(rec)

            if weighted > best_score:
                best_score = weighted
                best_text  = current_strategy

            if weighted >= QUALITY_THRESHOLD:
                break

            current_critique = feedback

        return RefineResult(
            final_strategy=best_text,
            final_score=best_score,
            cycles_completed=len(history),
            history=history,
            passed_threshold=best_score >= QUALITY_THRESHOLD,
        )

    # ─────────────────────────────────────────────────────────────────────

    async def _generate(
        self,
        situation: str,
        evidence_summary: str,
        case_law_context: str,
        attack_surface_summary: str,
        state: str,
        doc_mode: str,
        previous_strategy: str,
        previous_critique: str,
        cycle: int,
    ) -> str:
        action_label = (
            "petition for relief" if doc_mode == "petition"
            else "response and objection to the petition"
        )
        refine_block = ""
        if cycle > 1 and previous_strategy:
            refine_block = f"""
PREVIOUS STRATEGY DRAFT (cycle {cycle - 1}):
{previous_strategy[:3000]}

CRITIQUE — WEAKNESSES TO FIX THIS CYCLE:
{previous_critique[:2000]}

Produce an improved strategy that DIRECTLY addresses every critique above.
"""

        prompt = f"""You are the most formidable legal strategist in {state}.
Your task: synthesize an IRREFUTABLE, CONCRETE legal argument strategy for a {action_label}.

The strategy you produce will be injected verbatim into every document the system drafts.
It must be specific, factually grounded, and hardened against adversarial attack.
{refine_block}

CASE SITUATION:
{situation[:2000]}

EVIDENCE AVAILABLE:
{evidence_summary[:2000]}

RELEVANT CASE LAW:
{case_law_context[:2000]}

ATTACK SURFACE — WEAKNESSES WE MUST PRE-EMPT:
{attack_surface_summary[:2000]}

Produce a comprehensive LEGAL STRATEGY PLAN with these sections:

CASE THEORY (2-3 sentences max — the single overarching narrative):
<concise theory>

PRIMARY ARGUMENT (the single strongest, lead argument):
<argument with specific statute/case citations>

SUPPORTING ARGUMENTS (ranked by strength, each with evidence hook):
1. <argument + evidence + statute/case>
2. <argument + evidence + statute/case>
3. <argument + evidence + statute/case>
[add more as needed]

ATTACK PRE-EMPTION (for each identified weakness, our counter):
- [weakness type]: <our pre-emptive response>

KEY STATUTES (with section numbers):
- <statute § section>

KEY CASES (with citation):
- <Case Name, citation (court year)>

PRAYER / RELIEF SOUGHT:
<specific relief, numbered>

Write the complete strategy plan now. Be concrete — name specific facts, dates, amounts, and citations."""

        result = await self._llm.generate(
            prompt=prompt,
            task="legal_synthesis",
        )
        return (result.get("text") or "").strip()

    async def _critique(self, strategy_text: str):
        rubric_lines = "\n".join(
            f"  - {name} (weight {w}): {desc}"
            for name, w, desc in _RUBRIC
        )

        prompt = f"""You are a senior appellate judge evaluating a proposed legal strategy.
Score each criterion strictly on a 1-10 scale, then explain specific deficiencies.

RUBRIC:
{rubric_lines}

LEGAL STRATEGY TO EVALUATE:
{strategy_text[:4000]}

Return ONLY valid JSON:
{{
  "scores": {{
    "factual_support": <1-10>,
    "legal_authority": <1-10>,
    "argument_strength": <1-10>,
    "completeness": <1-10>,
    "procedural_soundness": <1-10>,
    "counter_resistance": <1-10>
  }},
  "feedback": "Specific weaknesses that must be fixed, referencing exact passages."
}}"""

        result = await self._llm.generate(
            prompt=prompt,
            task="classification",
            return_json=True,
        )

        try:
            data   = json.loads(result.get("text", "{}"))
            scores = {k: float(v) for k, v in data.get("scores", {}).items()}
            return scores, data.get("feedback", "")
        except (json.JSONDecodeError, TypeError, ValueError):
            return {name: 5.0 for name, _, _ in _RUBRIC}, "Parse error — no specific feedback."
