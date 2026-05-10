"""
Legal Strategy RAG — embedded knowledge base of winning court strategies.

Simulates a RAG retrieval system without requiring a vector database.
The knowledge base encodes patterns extracted from legal practice guides,
appellate decision analysis, and trial advocacy research.

The LLM acts as the "retriever" — given the case type and situation,
it selects the most relevant strategy patterns from the knowledge base
and formats them for injection into the working memory.

Knowledge base covers:
  Guardianship / Conservatorship
  Civil Rights / Constitutional
  Contract / Commercial
  Family Law
  Property / Eviction / Landlord-Tenant
  Employment / Discrimination
  Criminal Defense
  Administrative / Agency
  General Judicial Psychology
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

_KNOWLEDGE_BASE: Dict[str, Dict] = {

    "guardianship_conservatorship": {
        "judge_priorities": [
            "Protected person's autonomy and dignity — judges HATE unnecessary deprivation of rights",
            "Whether the petitioner's evidence is medical/clinical vs. purely anecdotal",
            "Whether less restrictive alternatives (e.g. power of attorney, representative payee) were tried first",
            "The financial motive of the petitioning party — guardianship is frequently abused for control",
            "The protected person's own expressed wishes and capacity to articulate them",
            "Whether the petitioner has a conflict of interest (inheritance, property access)",
        ],
        "winning_patterns": [
            "Establish concrete evidence of independent competent function: bank records, bills paid on time, "
            "tax returns filed, recent business transactions conducted without assistance",
            "Attack the medical evaluation: was it performed by the petitioner's hired expert? "
            "Was it a single snapshot or a longitudinal assessment? Did evaluator observe the person in their home environment?",
            "Invoke the constitutional dimension: involuntary guardianship is a deprivation of liberty "
            "under the 14th Amendment — due process requires the highest standard of proof",
            "Document the financial dispute: establish a timeline showing the guardianship petition "
            "followed a financial disagreement, inheritance claim, or property dispute",
            "Propose specific less restrictive alternatives with concrete implementation plans — "
            "judges cannot grant full guardianship if a lesser restriction would suffice",
            "Get a competing independent medical evaluation — judges give significant weight to "
            "conflicting expert opinions and will not rubber-stamp the petitioner's expert alone",
        ],
        "preemption_tactics": [
            "Petitioner will cite isolated incidents — counter with pattern evidence of competent behavior",
            "Petitioner's expert will testify to incapacity — demand cross on methodology, "
            "single-visit duration, whether they reviewed financial/behavioral history",
            "Petitioner will claim the protected person cannot manage finances — "
            "counter with years of tax returns, bank statements, investment accounts",
            "Petitioner will claim others corroborate incapacity — attack hearsay, "
            "bias, and the fact that these witnesses share a financial interest",
        ],
        "fatal_mistakes_to_avoid": [
            "Never concede any element of incapacity — partial concessions become full guardianship",
            "Never rely solely on the protected person's testimony — judges discount self-assessment",
            "Never let the financial dispute be characterized as irrelevant to capacity",
        ],
    },

    "civil_rights_constitutional": {
        "judge_priorities": [
            "Whether the constitutional violation is clearly established in binding precedent",
            "Whether the plaintiff has standing (injury-in-fact, causation, redressability)",
            "Whether qualified immunity bars the individual defendant claims",
            "The severity and documentation of the constitutional harm",
            "Whether administrative remedies were exhausted before suit",
        ],
        "winning_patterns": [
            "Lead with the most clearly established right — anchor every argument to a specific holding, "
            "not a general principle. Judges are reluctant to expand constitutional doctrine.",
            "Qualified immunity: find cases with materially similar facts showing the right was "
            "clearly established at the time of the violation",
            "Document the pattern, not just the incident — official policy or custom defeats "
            "qualified immunity and entity liability simultaneously",
            "Use the defendant's own records against them: internal memos, training materials, "
            "prior complaints, inspection reports — these establish notice",
        ],
        "preemption_tactics": [
            "Defendant will argue qualified immunity — have your 'clearly established' cases ready "
            "with direct fact-to-fact comparison",
            "Defendant will argue no policy or custom — pre-empt with pattern evidence "
            "of multiple similar incidents before yours",
        ],
        "fatal_mistakes_to_avoid": [
            "Never rely on a right that was only established in other circuits — "
            "binding precedent must be from the same circuit or SCOTUS",
        ],
    },

    "contract_commercial": {
        "judge_priorities": [
            "The four corners of the written agreement — judges resist excursions beyond the contract",
            "The parties' course of dealing and course of performance",
            "Whether the breach was material (excusing performance) or minor (only damages)",
            "Mitigation of damages — failure to mitigate is a complete defense to excess losses",
            "The specific damages calculation — vague 'I lost money' claims fail without proof",
        ],
        "winning_patterns": [
            "Always anchor to contract language first, then extrinsic evidence only if ambiguous",
            "Compute damages with a specific number, supported by invoices, market rates, "
            "or expert testimony — 'reasonable certainty' is the standard",
            "Show the breach was material: it went to the essence of the bargain, "
            "not a technical or trivial deviation",
            "Document your mitigation efforts — show you tried to reduce losses",
        ],
        "preemption_tactics": [
            "Defendant will argue no breach — have the specific contract clause and "
            "the specific act of non-performance side by side",
            "Defendant will argue your damages are speculative — prepare three independent "
            "damage calculations using different methodologies that converge on the same number",
        ],
        "fatal_mistakes_to_avoid": [
            "Never claim damages you cannot document with specific numbers",
            "Never ignore a limitation of liability clause — address it head-on",
        ],
    },

    "property_eviction_landlord_tenant": {
        "judge_priorities": [
            "Strict compliance with statutory notice requirements — procedural defects are often dispositive",
            "Whether the landlord followed the exact statutory notice timeline and form",
            "Whether habitability issues defeat the landlord's claim for rent",
            "The tenant's payment history and good-faith efforts to pay",
        ],
        "winning_patterns": [
            "Attack notice: most evictions fail on procedural grounds — count the days carefully, "
            "check the form, check the service method",
            "Habitability defense: document every repair request in writing, "
            "every inspection report, every communication about the defect",
            "Retaliatory eviction: timeline the eviction notice against any complaint "
            "you filed with housing authorities — statutory presumption of retaliation if "
            "eviction follows within 90 days of a complaint in most states",
        ],
        "preemption_tactics": [
            "Landlord will produce a lease — examine it for illegal provisions that void the landlord's claims",
            "Landlord will claim nonpayment — show any payments made and dispute "
            "whether rent was properly due given habitability issues",
        ],
        "fatal_mistakes_to_avoid": [
            "Never miss a statutory response deadline — courts strictly enforce them in eviction proceedings",
        ],
    },

    "employment_discrimination": {
        "judge_priorities": [
            "Whether plaintiff can establish prima facie case under McDonnell Douglas framework",
            "Whether the employer's stated reason is a pretext — and the specific evidence of pretext",
            "Comparator evidence — similarly situated employees treated differently",
            "Direct evidence vs. circumstantial evidence (different legal standards apply)",
            "Whether the plaintiff filed a timely EEOC charge before suing",
        ],
        "winning_patterns": [
            "Comparator analysis: identify specific, named comparators who are similarly situated "
            "but were treated more favorably — vague 'others were treated better' claims fail",
            "Temporal proximity: if adverse action follows a protected activity within a few weeks, "
            "this is strong circumstantial evidence of retaliation",
            "Document the employer's shifting explanations for the adverse action — "
            "inconsistent explanations are powerful pretext evidence",
        ],
        "preemption_tactics": [
            "Employer will argue legitimate non-discriminatory reason — have your pretext evidence ready",
            "Employer will attack comparators as not 'similarly situated' — "
            "anchor to objective criteria: same supervisor, same job duties, same misconduct if any",
        ],
        "fatal_mistakes_to_avoid": [
            "Never miss the EEOC charge deadline — it is jurisdictional in federal court",
            "Never compare yourself to employees in different departments or under different supervisors",
        ],
    },

    "general_judicial_psychology": {
        "universal_judge_priorities": [
            "CREDIBILITY: Judges decide cases on who they believe. Evidence of honesty and candor "
            "in the party matters enormously — never overstate a claim.",
            "PROPORTIONALITY: The relief sought must be proportional to the harm proven. "
            "Asking for too much signals bad faith and sours the judge on everything.",
            "PROCEDURAL COMPLIANCE: Judges are sticklers for following their own rules. "
            "Filings that cite wrong rules, miss deadlines, or ignore local rules create "
            "an immediate negative impression that is hard to overcome.",
            "EFFICIENCY: Judges are overworked. Arguments that waste time with irrelevant "
            "facts or digressions lose judicial patience quickly.",
            "NARRATIVE COHERENCE: Judges are human — they are more persuaded by a story "
            "that makes sense than by a list of legal points. The facts must tell a "
            "coherent narrative where your position is the obvious, just result.",
            "WORST FIRST: In bench trials, judges read everything before the hearing. "
            "Lead with your strongest argument — if the judge is already convinced on "
            "argument 1, arguments 2-5 reinforce rather than having to carry the day alone.",
        ],
        "argument_structure_principles": [
            "IRAC at every level: Issue → Rule → Application → Conclusion. "
            "Judges are trained on IRAC and find it easiest to follow.",
            "One argument per section header — mixing arguments in a single section "
            "dilutes them both.",
            "Cite to the record obsessively — every factual claim must have a record cite "
            "(exhibit, deposition page, affidavit paragraph). Unsupported factual claims "
            "are not just weak — they damage your credibility on everything else.",
            "Anticipate and address the hardest argument against you. A brief that "
            "ignores the obvious counter-argument looks incomplete and raises suspicion.",
        ],
        "tone_and_delivery": [
            "Respectful, measured, and professional — never attack the opposing party personally",
            "Confident but not arrogant — overconfidence triggers skepticism",
            "Show you understand the opposing argument before you destroy it — "
            "'While respondent argues X, this fails because...' is more persuasive "
            "than ignoring their argument",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# RAG RETRIEVER
# ═══════════════════════════════════════════════════════════════════════════════

class LegalRAG:
    """
    Retrieves relevant legal strategy patterns from the embedded knowledge base.

    The LLM is used as the semantic retriever — given the case description,
    it identifies which knowledge base categories apply and selects the most
    relevant patterns, ranked by importance.
    """

    def __init__(self, llm_client: Any):
        self._llm = llm_client

    async def retrieve(
        self,
        situation: str,
        case_type_hint: str = "",
        top_k: int = 20,
    ) -> str:
        """
        Retrieve the most relevant strategy patterns for this case.

        Returns a formatted string ready for injection into working memory.
        """

        # Step 1: classify the case type
        case_type = await self._classify(situation, case_type_hint)

        # Step 2: pull patterns from the knowledge base
        patterns = self._pull_patterns(case_type)

        # Step 3: LLM ranks and filters patterns by relevance to THIS specific case
        ranked = await self._rank_and_filter(situation, patterns, top_k)

        return ranked

    async def _classify(self, situation: str, hint: str) -> str:
        """Identify which knowledge base category best matches this case."""
        categories = list(_KNOWLEDGE_BASE.keys())

        prompt = f"""Classify this legal case into ONE of these categories:
{chr(10).join(f'  - {c}' for c in categories)}

CASE SITUATION:
{situation[:800]}

HINT (if provided): {hint}

Return ONLY the exact category name from the list above.
If multiple apply, choose the PRIMARY one.
If none match closely, return "general_judicial_psychology"."""

        result = await self._llm.generate(prompt=prompt, task="classification")
        raw    = (result.get("text") or "").strip().lower().replace(" ", "_")

        # Find closest match
        for cat in categories:
            if cat in raw or raw in cat:
                return cat
        return "general_judicial_psychology"

    def _pull_patterns(self, case_type: str) -> Dict:
        """Pull patterns from the knowledge base for the given case type."""
        # Always include general judicial psychology
        general = _KNOWLEDGE_BASE.get("general_judicial_psychology", {})
        specific = _KNOWLEDGE_BASE.get(case_type, {})

        return {
            "case_type": case_type,
            "general": general,
            "specific": specific,
        }

    async def _rank_and_filter(
        self, situation: str, patterns: Dict, top_k: int
    ) -> str:
        """Use LLM to rank the retrieved patterns by relevance to this specific case."""

        general   = patterns.get("general", {})
        specific  = patterns.get("specific", {})
        case_type = patterns.get("case_type", "unknown")

        # Build the raw knowledge block
        raw_lines = [
            f"=== KNOWLEDGE BASE: {case_type.upper().replace('_', ' ')} ===",
            "",
        ]

        if specific.get("judge_priorities"):
            raw_lines.append("WHAT THE JUDGE CARES ABOUT MOST:")
            for p in specific["judge_priorities"]:
                raw_lines.append(f"  • {p}")
            raw_lines.append("")

        if specific.get("winning_patterns"):
            raw_lines.append("WINNING PATTERNS IN THIS TYPE OF CASE:")
            for p in specific["winning_patterns"]:
                raw_lines.append(f"  • {p}")
            raw_lines.append("")

        if specific.get("preemption_tactics"):
            raw_lines.append("WHAT OPPOSING COUNSEL WILL ARGUE AND HOW TO PREEMPT IT:")
            for p in specific["preemption_tactics"]:
                raw_lines.append(f"  • {p}")
            raw_lines.append("")

        if specific.get("fatal_mistakes_to_avoid"):
            raw_lines.append("FATAL MISTAKES TO AVOID:")
            for p in specific["fatal_mistakes_to_avoid"]:
                raw_lines.append(f"  • {p}")
            raw_lines.append("")

        raw_lines.append("=== UNIVERSAL JUDICIAL PSYCHOLOGY ===")
        raw_lines.append("")
        for p in general.get("universal_judge_priorities", []):
            raw_lines.append(f"  • {p}")
        raw_lines.append("")
        for p in general.get("argument_structure_principles", []):
            raw_lines.append(f"  • {p}")

        raw_knowledge = "\n".join(raw_lines)

        # Ask LLM to filter and rank by relevance to THIS specific case
        prompt = f"""You are a senior trial strategist reviewing a legal knowledge base.
Select and rank the patterns below that are MOST RELEVANT to this specific case.
Discard anything that does not apply. Add a one-line note after each item explaining
WHY it is specifically relevant to the facts of this case.

CASE SITUATION:
{situation[:800]}

KNOWLEDGE BASE TO FILTER:
{raw_knowledge}

Return the top {top_k} most relevant items, each on its own line, formatted:
  PRIORITY [1-{top_k}]: <pattern text> — RELEVANCE: <why this applies here>

Return ONLY the ranked list, no preamble."""

        result = await self._llm.generate(prompt=prompt, task="summarization")
        return (result.get("text") or raw_knowledge).strip()
