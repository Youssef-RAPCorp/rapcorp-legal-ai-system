"""
Legal Thinking Engine — pre-generation strategy phase.

Orchestrates:
  1. Attack Surface Agent  — adversarial analysis of both sides
  2. Legal Self-Refiner   — Generate → Critique → Refine argument strategy

Output: a LegalStrategyPlan injected into every document generation prompt,
ensuring all documents express a single, hardened, irrefutable legal position.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.thinking.attack_surface_agent import AttackSurfaceAgent, AttackSurfaceReport
from src.thinking.legal_self_refine import LegalSelfRefiner, RefineResult


@dataclass
class LegalStrategyPlan:
    """Fully synthesized legal strategy — fed into every document prompt."""
    case_theory:            str
    primary_argument:       str
    supporting_arguments:   List[str]
    counter_strategies:     List[str]      # pre-emptions of our weaknesses
    their_weak_points:      List[str]      # angles to press against them
    key_statutes:           List[str]
    key_cases:              List[str]
    prayer_for_relief:      str
    argument_score:         float
    refinement_cycles:      int
    attack_surface_report:  Optional[AttackSurfaceReport] = None

    def as_prompt_block(self) -> str:
        """Render the strategy as a block for injection into any prompt."""
        lines = [
            "═══ LEGAL STRATEGY PLAN (DO NOT DEVIATE FROM THIS) ═══",
            "",
            "CASE THEORY:",
            self.case_theory,
            "",
            "PRIMARY ARGUMENT (lead with this in every document):",
            self.primary_argument,
            "",
            "SUPPORTING ARGUMENTS (use ALL of these):",
        ]
        for i, arg in enumerate(self.supporting_arguments, 1):
            lines.append(f"  {i}. {arg}")
        lines += [
            "",
            "ATTACK PRE-EMPTIONS (address EACH of these before opposing counsel raises them):",
        ]
        for ctr in self.counter_strategies:
            lines.append(f"  • {ctr}")
        lines += [
            "",
            "THEIR WEAK POINTS (press these aggressively):",
        ]
        for wp in self.their_weak_points:
            lines.append(f"  • {wp}")
        if self.key_statutes:
            lines += ["", "MANDATORY STATUTES (cite every one by exact section):"]
            for s in self.key_statutes:
                lines.append(f"  - {s}")
        if self.key_cases:
            lines += ["", "MANDATORY CASE CITATIONS (cite every one by name and citation):"]
            for c in self.key_cases:
                lines.append(f"  - {c}")
        if self.prayer_for_relief:
            lines += ["", "PRAYER FOR RELIEF:", self.prayer_for_relief]
        lines += [
            "",
            f"[Strategy quality score: {self.argument_score:.1f}/10 after {self.refinement_cycles} refinement cycles]",
            "═══════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


class LegalThinkingEngine:
    """
    Pre-generation thinking phase.

    Usage:
        engine = LegalThinkingEngine(llm_client)
        plan = await engine.think(
            situation, evidence_summary, case_law_context,
            case_documents_context, state, doc_mode
        )
    """

    def __init__(self, llm_client: Any):
        self._llm     = llm_client
        self._attack  = AttackSurfaceAgent(llm_client)
        self._refiner = LegalSelfRefiner(llm_client)

    async def think(
        self,
        situation: str,
        evidence_summary: str,
        case_law_context: str,
        case_documents_context: str,
        state: str,
        doc_mode: str = "petition",
    ) -> LegalStrategyPlan:
        """
        Run the full thinking pipeline and return a concrete strategy plan.

        Steps:
          1. Attack surface analysis (our weaknesses + their weaknesses)
          2. Self-refine loop: generate → critique → refine (up to 3 cycles)
          3. Parse the refined strategy into a structured LegalStrategyPlan
        """

        print("  [Thinking] Phase 1: Attack surface analysis...")
        report: AttackSurfaceReport = await self._attack.analyze(
            situation=situation,
            evidence_summary=evidence_summary,
            case_law_context=case_law_context,
            case_documents_context=case_documents_context,
            state=state,
            doc_mode=doc_mode,
        )
        print(f"    Found {len(report.our_weaknesses)} weaknesses in our case, "
              f"{len(report.their_weaknesses)} in theirs "
              f"({report.critical_count} critical).")

        # Build compact attack surface summary for refiner
        attack_summary = self._format_attack_surface(report)

        print("  [Thinking] Phase 2: Self-refine argument strategy...")
        refine_result: RefineResult = await self._refiner.refine(
            situation=situation,
            evidence_summary=evidence_summary,
            case_law_context=case_law_context,
            attack_surface_summary=attack_summary,
            state=state,
            doc_mode=doc_mode,
        )
        score = refine_result.final_score
        cycles = refine_result.cycles_completed
        passed = "PASS" if refine_result.passed_threshold else "BEST EFFORT"
        print(f"    Strategy score: {score:.1f}/10 ({passed}) after {cycles} cycle(s).")

        # Parse the refined strategy text into structured fields
        plan = self._parse_strategy(
            strategy_text=refine_result.final_strategy,
            report=report,
            score=score,
            cycles=cycles,
        )

        return plan

    # ─────────────────────────────────────────────────────────────────────

    def _format_attack_surface(self, report: AttackSurfaceReport) -> str:
        lines = ["OUR WEAKNESSES (must pre-empt):"]
        for v in report.our_weaknesses[:8]:
            sev = f"{v.severity:.1f}"
            lines.append(f"  [{sev}] {v.vector_type.value}: {v.description[:200]}")
            if v.counter_strategy:
                lines.append(f"        → Counter: {v.counter_strategy[:150]}")
        lines += ["", "THEIR WEAKNESSES (press these):"]
        for v in report.their_weaknesses[:8]:
            sev = f"{v.severity:.1f}"
            lines.append(f"  [{sev}] {v.vector_type.value}: {v.description[:200]}")
            if v.counter_strategy:
                lines.append(f"        → Attack: {v.counter_strategy[:150]}")
        return "\n".join(lines)

    def _parse_strategy(
        self,
        strategy_text: str,
        report: AttackSurfaceReport,
        score: float,
        cycles: int,
    ) -> LegalStrategyPlan:
        """Extract structured fields from the free-text strategy."""

        def _extract_section(text: str, header: str) -> str:
            """Pull the text between `header:` and the next ALL-CAPS header."""
            idx = text.upper().find(header.upper())
            if idx == -1:
                return ""
            after = text[idx + len(header):].lstrip(": \n")
            next_header = _next_header_pos(after)
            return after[:next_header].strip() if next_header else after.strip()

        def _next_header_pos(text: str) -> int:
            """Find start of the next section header (ALL CAPS line)."""
            import re
            for m in re.finditer(r'\n([A-Z][A-Z\s]{4,}):', text):
                return m.start()
            return len(text)

        def _extract_list_section(text: str, header: str) -> List[str]:
            raw = _extract_section(text, header)
            if not raw:
                return []
            items = []
            for line in raw.split("\n"):
                line = line.strip().lstrip("•-–*123456789. ")
                if line:
                    items.append(line)
            return items

        # Pull key sections
        case_theory   = _extract_section(strategy_text, "CASE THEORY")
        primary_arg   = _extract_section(strategy_text, "PRIMARY ARGUMENT")
        prayer        = _extract_section(strategy_text, "PRAYER")
        support_args  = _extract_list_section(strategy_text, "SUPPORTING ARGUMENTS")
        key_statutes  = _extract_list_section(strategy_text, "KEY STATUTES")
        key_cases     = _extract_list_section(strategy_text, "KEY CASES")

        # Counter-strategies from attack surface
        counter_strategies = [
            v.counter_strategy for v in report.our_weaknesses
            if v.counter_strategy
        ][:8]

        # Their weak points
        their_weak_points = [
            f"{v.vector_type.value}: {v.description[:150]}"
            for v in report.their_weaknesses[:6]
        ]

        # Fallback: if parsing found nothing, use the full strategy text
        if not case_theory and not primary_arg:
            case_theory = strategy_text[:500]
            primary_arg = strategy_text[:1000]

        return LegalStrategyPlan(
            case_theory=case_theory[:800],
            primary_argument=primary_arg[:1500],
            supporting_arguments=support_args[:10],
            counter_strategies=counter_strategies,
            their_weak_points=their_weak_points,
            key_statutes=key_statutes[:15],
            key_cases=key_cases[:15],
            prayer_for_relief=prayer[:800],
            argument_score=score,
            refinement_cycles=cycles,
            attack_surface_report=report,
        )
