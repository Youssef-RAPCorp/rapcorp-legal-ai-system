"""
Attack Surface Agent — adversarial pre-generation analysis.

Identifies every weakness in our case that opposing counsel could exploit, AND
every weakness in the opposing party's case that we can attack.  Output feeds
directly into the strategy synthesis step.

Legal attack vectors (ported from Research-Agent peer_review.py):
  FACTUAL_WEAKNESS       — facts that are vague, unverifiable, or contradicted
  LEGAL_GAP              — elements of the cause of action we cannot prove
  PROCEDURAL_FLAW        — missed deadlines, wrong court, improper service
  STATUTORY_INCONSISTENCY— our statutory theory conflicts with controlling law
  ALTERNATIVE_DEFENSE    — defenses the opponent could raise against us
  CREDIBILITY_ATTACK     — witnesses / documents the opponent could impeach
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class LegalVector(str, Enum):
    FACTUAL_WEAKNESS        = "factual_weakness"
    LEGAL_GAP               = "legal_gap"
    PROCEDURAL_FLAW         = "procedural_flaw"
    STATUTORY_INCONSISTENCY = "statutory_inconsistency"
    ALTERNATIVE_DEFENSE     = "alternative_defense"
    CREDIBILITY_ATTACK      = "credibility_attack"


@dataclass
class LegalAttackVector:
    vector_type: LegalVector
    severity: float           # 0.0 – 1.0
    target: str               # "our_case" | "their_case"
    description: str
    specific_evidence: List[str] = field(default_factory=list)
    counter_strategy: str = ""


@dataclass
class AttackSurfaceReport:
    our_weaknesses: List[LegalAttackVector]
    their_weaknesses: List[LegalAttackVector]
    critical_count: int
    overall_exposure: float   # 0.0 – 1.0; lower is better for us
    top_counter_strategies: List[str]
    raw_json: str = ""


class AttackSurfaceAgent:
    """
    Adversarial agent that stress-tests both sides of a legal dispute
    before any document is drafted.
    """

    _VECTOR_DESCRIPTIONS = {
        LegalVector.FACTUAL_WEAKNESS:        "facts that are vague, unverifiable, or internally contradicted",
        LegalVector.LEGAL_GAP:               "required legal elements we cannot satisfy with current evidence",
        LegalVector.PROCEDURAL_FLAW:         "missed deadlines, wrong court, improper service, or procedural defects",
        LegalVector.STATUTORY_INCONSISTENCY: "statutory theories that conflict with controlling case law",
        LegalVector.ALTERNATIVE_DEFENSE:     "affirmative defenses the opponent will raise",
        LegalVector.CREDIBILITY_ATTACK:      "witnesses or documents the opponent can impeach",
    }

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def analyze(
        self,
        situation: str,
        evidence_summary: str,
        case_law_context: str,
        case_documents_context: str,
        state: str,
        doc_mode: str = "petition",
    ) -> AttackSurfaceReport:
        """Run full adversarial analysis on both sides."""

        our_weaknesses   = await self._attack_our_case(
            situation, evidence_summary, case_law_context,
            case_documents_context, state, doc_mode
        )
        their_weaknesses = await self._attack_their_case(
            situation, evidence_summary, case_law_context,
            case_documents_context, state, doc_mode
        )

        critical = sum(1 for w in our_weaknesses if w.severity >= 0.7)
        avg_sev  = (
            sum(w.severity for w in our_weaknesses) / len(our_weaknesses)
            if our_weaknesses else 0.0
        )

        top_counters: List[str] = []
        for w in sorted(our_weaknesses, key=lambda x: x.severity, reverse=True)[:5]:
            if w.counter_strategy:
                top_counters.append(f"[{w.vector_type.value}] {w.counter_strategy}")

        return AttackSurfaceReport(
            our_weaknesses=our_weaknesses,
            their_weaknesses=their_weaknesses,
            critical_count=critical,
            overall_exposure=avg_sev,
            top_counter_strategies=top_counters,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────

    async def _attack_our_case(
        self,
        situation: str,
        evidence_summary: str,
        case_law_context: str,
        case_documents_context: str,
        state: str,
        doc_mode: str,
    ) -> List[LegalAttackVector]:
        """Pretend to be opposing counsel and attack our position."""

        mode_label = "respondent defending against a petition" if doc_mode == "petition" else "opposing party"

        prompt = f"""You are the most aggressive, analytically precise opposing attorney in {state}.
Your job: identify EVERY weakness in the CLIENT'S case below that you could exploit at trial.
Be relentlessly adversarial. Assume you have infinite resources.

CASE SITUATION (our client's perspective):
{situation[:2000]}

EVIDENCE SUMMARY:
{evidence_summary[:2000]}

RELEVANT CASE LAW:
{case_law_context[:1500]}

PRIOR CASE DOCUMENTS:
{case_documents_context[:1500]}

For each attack vector, produce a JSON object in this array:
{{
  "attack_vectors": [
    {{
      "vector_type": "factual_weakness | legal_gap | procedural_flaw | statutory_inconsistency | alternative_defense | credibility_attack",
      "severity": 0.0-1.0,
      "description": "Exactly what the weakness is and how you would exploit it",
      "specific_evidence": ["The specific fact or document that creates the problem"],
      "counter_strategy": "How OUR client should pre-empt or mitigate this attack"
    }}
  ]
}}

Return ONLY valid JSON. Identify at minimum 6 attack vectors, more if they exist."""

        result = await self._llm.generate(
            prompt=prompt,
            task="legal_synthesis",
            return_json=True,
        )

        return self._parse_vectors(result.get("text", "{}"), target="our_case")

    async def _attack_their_case(
        self,
        situation: str,
        evidence_summary: str,
        case_law_context: str,
        case_documents_context: str,
        state: str,
        doc_mode: str,
    ) -> List[LegalAttackVector]:
        """Identify weaknesses in the opposing party's position."""

        prompt = f"""You are a senior litigation strategist representing our client in {state}.
Your job: identify EVERY exploitable weakness in the OPPOSING PARTY'S case.
These are the attack angles we will press at trial.

OUR CLIENT'S SITUATION:
{situation[:2000]}

OUR EVIDENCE:
{evidence_summary[:2000]}

RELEVANT CASE LAW:
{case_law_context[:1500]}

OPPOSING PARTY'S DOCUMENTS (if any):
{case_documents_context[:1500]}

For each weakness in the opposing party's case, produce JSON:
{{
  "attack_vectors": [
    {{
      "vector_type": "factual_weakness | legal_gap | procedural_flaw | statutory_inconsistency | alternative_defense | credibility_attack",
      "severity": 0.0-1.0,
      "description": "The specific weakness in THEIR case and exactly how we press it",
      "specific_evidence": ["The fact, document, or gap that exposes their weakness"],
      "counter_strategy": "The argument or motion we should file to exploit this"
    }}
  ]
}}

Return ONLY valid JSON. Find at minimum 5 weaknesses in their position."""

        result = await self._llm.generate(
            prompt=prompt,
            task="legal_synthesis",
            return_json=True,
        )

        return self._parse_vectors(result.get("text", "{}"), target="their_case")

    def _parse_vectors(self, raw: str, target: str) -> List[LegalAttackVector]:
        vectors: List[LegalAttackVector] = []
        try:
            data = json.loads(raw)
            items = data.get("attack_vectors", [])
            for item in items:
                vtype_str = item.get("vector_type", "factual_weakness")
                try:
                    vtype = LegalVector(vtype_str)
                except ValueError:
                    vtype = LegalVector.FACTUAL_WEAKNESS
                vectors.append(LegalAttackVector(
                    vector_type=vtype,
                    severity=float(item.get("severity", 0.5)),
                    target=target,
                    description=item.get("description", ""),
                    specific_evidence=item.get("specific_evidence", []),
                    counter_strategy=item.get("counter_strategy", ""),
                ))
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return sorted(vectors, key=lambda v: v.severity, reverse=True)
