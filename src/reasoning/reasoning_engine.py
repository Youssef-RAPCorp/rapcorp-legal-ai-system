"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    LEGAL REASONING ENGINE                                      ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Advanced Jurisprudential AI Reasoning:                                       ║
║  • IRAC Framework (Issue, Rule, Application, Conclusion)                      ║
║  • CREAC Framework (Conclusion, Rule, Explanation, Application, Conclusion)   ║
║  • Multi-Step Legal Reasoning Chains                                          ║
║  • Argument Construction & Counterargument Generation                         ║
║  • Precedent Analysis & Distinguishing                                        ║
║  • Damages Calculation Engine                                                 ║
║  • Statute of Limitations Calculator                                          ║
║  • Legal Syllogism Validator                                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import json

from configs.config import LegalAIConfig, USState, LegalDomain, GeminiModel


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ReasoningFramework(Enum):
    """Legal reasoning frameworks."""
    IRAC = "irac"  # Issue, Rule, Application, Conclusion
    CREAC = "creac"  # Conclusion, Rule, Explanation, Application, Conclusion
    TREAT = "treat"  # Thesis, Rule, Explanation, Application, Thesis
    CRAC = "crac"  # Conclusion, Rule, Application, Conclusion


class DamageType(Enum):
    """Types of legal damages."""
    COMPENSATORY = "compensatory"
    CONSEQUENTIAL = "consequential"
    PUNITIVE = "punitive"
    NOMINAL = "nominal"
    LIQUIDATED = "liquidated"
    STATUTORY = "statutory"
    TREBLE = "treble"
    EMOTIONAL_DISTRESS = "emotional_distress"


@dataclass
class IRACAnalysis:
    """IRAC analysis structure."""
    issue: str
    rule: str
    application: str
    conclusion: str
    confidence: float = 0.0
    supporting_cases: List[str] = field(default_factory=list)
    counterarguments: List[str] = field(default_factory=list)


@dataclass
class CREACAnalysis:
    """CREAC analysis structure."""
    conclusion_preview: str
    rule_statement: str
    rule_explanation: str
    application: str
    conclusion_final: str
    confidence: float = 0.0
    supporting_authorities: List[str] = field(default_factory=list)


@dataclass
class LegalArgument:
    """A structured legal argument."""
    claim: str
    grounds: List[str]
    warrant: str
    backing: List[str]
    qualifier: str
    rebuttal: Optional[str] = None
    strength: float = 0.0


@dataclass
class DamagesCalculation:
    """Result of damages calculation."""
    damage_type: DamageType
    base_amount: Decimal
    adjustments: Dict[str, Decimal] = field(default_factory=dict)
    multiplier: float = 1.0
    total_amount: Decimal = Decimal("0")
    calculation_notes: List[str] = field(default_factory=list)
    jurisdiction_rules: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_total(self) -> Decimal:
        """Calculate the total damages."""
        total = self.base_amount
        
        # Apply adjustments
        for name, amount in self.adjustments.items():
            total += amount
        
        # Apply multiplier
        total = total * Decimal(str(self.multiplier))
        
        self.total_amount = total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return self.total_amount


@dataclass
class SOLCalculation:
    """Statute of limitations calculation."""
    state: USState
    claim_type: str
    accrual_date: datetime
    limitation_years: int
    tolling_days: int = 0
    deadline: datetime = None
    is_expired: bool = False
    days_remaining: int = 0
    
    def calculate(self) -> None:
        """Calculate the deadline and status."""
        # Base deadline
        self.deadline = self.accrual_date.replace(
            year=self.accrual_date.year + self.limitation_years
        )
        
        # Add tolling
        self.deadline += timedelta(days=self.tolling_days)
        
        # Check status
        now = datetime.utcnow()
        self.is_expired = now > self.deadline
        self.days_remaining = max(0, (self.deadline - now).days)


# ═══════════════════════════════════════════════════════════════════════════════
# REASONING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LegalReasoningEngine:
    """
    Advanced legal reasoning engine using Gemini Pro for complex analysis.
    
    Capabilities:
    - IRAC/CREAC structured analysis
    - Argument construction
    - Counterargument generation
    - Damages calculation
    - Statute of limitations tracking
    - Legal syllogism validation
    """
    
    def __init__(self, config: LegalAIConfig, llm_client: Any = None):
        self.config = config
        self.llm = llm_client
        
        # Default to Pro model for reasoning tasks
        self.model = GeminiModel.PRO.value
    
    # ─── IRAC Analysis ───
    
    async def analyze_irac(
        self,
        issue: str,
        facts: Dict[str, Any],
        jurisdiction: USState = USState.FEDERAL,
        relevant_law: Optional[List[str]] = None
    ) -> IRACAnalysis:
        """
        Perform IRAC analysis on a legal issue.
        
        Args:
            issue: The legal issue to analyze
            facts: Relevant facts of the case
            jurisdiction: The jurisdiction
            relevant_law: Optional list of relevant authorities
        
        Returns:
            Structured IRAC analysis
        """
        prompt = f"""
        Perform a comprehensive IRAC analysis for the following legal issue.
        
        ISSUE: {issue}
        
        FACTS:
        {json.dumps(facts, indent=2)}
        
        JURISDICTION: {jurisdiction.value}
        
        RELEVANT AUTHORITIES:
        {json.dumps(relevant_law or [], indent=2)}
        
        Provide your analysis in the following JSON format:
        {{
            "issue": "Clear statement of the legal issue in question form",
            "rule": "Statement of the applicable legal rule(s) with citations",
            "application": "Detailed application of the rule to the facts",
            "conclusion": "Clear conclusion answering the issue",
            "confidence": 0.0-1.0,
            "supporting_cases": ["case1", "case2"],
            "counterarguments": ["potential counterargument 1", "potential counterargument 2"]
        }}
        """
        
        if self.llm:
            result = await self.llm.generate(
                prompt=prompt,
                task="irac_analysis",
                model_override=self.model,
                return_json=True
            )
            
            try:
                data = json.loads(result["text"])
                return IRACAnalysis(**data)
            except (json.JSONDecodeError, KeyError) as e:
                # Return with raw text if parsing fails
                return IRACAnalysis(
                    issue=issue,
                    rule="Unable to parse structured response",
                    application=result.get("text", ""),
                    conclusion="Analysis incomplete",
                    confidence=0.0
                )
        
        # Mock response if no LLM
        return IRACAnalysis(
            issue=issue,
            rule="[Rule would be generated by LLM]",
            application="[Application would be generated by LLM]",
            conclusion="[Conclusion would be generated by LLM]",
            confidence=0.0
        )
    
    # ─── CREAC Analysis ───
    
    async def analyze_creac(
        self,
        issue: str,
        position: str,
        facts: Dict[str, Any],
        jurisdiction: USState = USState.FEDERAL
    ) -> CREACAnalysis:
        """
        Perform CREAC analysis (persuasive writing framework).
        
        Args:
            issue: The legal issue
            position: The position to advocate
            facts: Case facts
            jurisdiction: The jurisdiction
        
        Returns:
            Structured CREAC analysis
        """
        prompt = f"""
        Perform a CREAC analysis to persuasively argue the following position.
        
        ISSUE: {issue}
        POSITION TO ADVOCATE: {position}
        JURISDICTION: {jurisdiction.value}
        
        FACTS:
        {json.dumps(facts, indent=2)}
        
        Provide your analysis in JSON format:
        {{
            "conclusion_preview": "Bold assertion of your conclusion upfront",
            "rule_statement": "Statement of the governing legal rule",
            "rule_explanation": "Explanation of how courts have applied this rule",
            "application": "Application of rule to these specific facts",
            "conclusion_final": "Restatement of conclusion with call to action",
            "confidence": 0.0-1.0,
            "supporting_authorities": ["authority1", "authority2"]
        }}
        """
        
        if self.llm:
            result = await self.llm.generate(
                prompt=prompt,
                task="creac_analysis",
                model_override=self.model,
                return_json=True
            )
            
            try:
                data = json.loads(result["text"])
                return CREACAnalysis(**data)
            except (json.JSONDecodeError, KeyError):
                return CREACAnalysis(
                    conclusion_preview=position,
                    rule_statement="Unable to parse",
                    rule_explanation=result.get("text", ""),
                    application="",
                    conclusion_final="",
                    confidence=0.0
                )
        
        return CREACAnalysis(
            conclusion_preview=position,
            rule_statement="[Would be generated by LLM]",
            rule_explanation="[Would be generated by LLM]",
            application="[Would be generated by LLM]",
            conclusion_final="[Would be generated by LLM]",
            confidence=0.0
        )
    
    # ─── Argument Construction ───
    
    async def construct_argument(
        self,
        claim: str,
        facts: Dict[str, Any],
        supporting_authorities: List[str],
        jurisdiction: USState = USState.FEDERAL
    ) -> LegalArgument:
        """
        Construct a structured legal argument using the Toulmin model.
        
        Args:
            claim: The main claim to argue
            facts: Supporting facts
            supporting_authorities: Cases/statutes supporting the claim
            jurisdiction: The jurisdiction
        
        Returns:
            Structured legal argument
        """
        prompt = f"""
        Construct a legal argument using the Toulmin model for the following claim.
        
        CLAIM: {claim}
        JURISDICTION: {jurisdiction.value}
        
        FACTS:
        {json.dumps(facts, indent=2)}
        
        SUPPORTING AUTHORITIES:
        {json.dumps(supporting_authorities, indent=2)}
        
        Return in JSON format:
        {{
            "claim": "The main assertion",
            "grounds": ["factual grounds supporting the claim"],
            "warrant": "Legal principle connecting grounds to claim",
            "backing": ["authorities backing the warrant"],
            "qualifier": "degree of certainty (e.g., 'likely', 'certainly')",
            "rebuttal": "acknowledgment of potential counterarguments",
            "strength": 0.0-1.0
        }}
        """
        
        if self.llm:
            result = await self.llm.generate(
                prompt=prompt,
                task="argument_construction",
                model_override=self.model,
                return_json=True
            )
            
            try:
                data = json.loads(result["text"])
                return LegalArgument(**data)
            except (json.JSONDecodeError, KeyError):
                return LegalArgument(
                    claim=claim,
                    grounds=[],
                    warrant="Unable to parse",
                    backing=[],
                    qualifier="uncertain",
                    strength=0.0
                )
        
        return LegalArgument(
            claim=claim,
            grounds=["[Would be generated by LLM]"],
            warrant="[Would be generated by LLM]",
            backing=supporting_authorities,
            qualifier="[Would be generated by LLM]",
            strength=0.0
        )
    
    # ─── Counterargument Generation ───
    
    async def generate_counterarguments(
        self,
        argument: LegalArgument,
        opposing_position: str
    ) -> List[LegalArgument]:
        """
        Generate counterarguments to a legal argument.
        
        Args:
            argument: The argument to counter
            opposing_position: The opposing party's position
        
        Returns:
            List of counterarguments
        """
        prompt = f"""
        Generate counterarguments to the following legal argument.
        
        ORIGINAL ARGUMENT:
        Claim: {argument.claim}
        Grounds: {json.dumps(argument.grounds)}
        Warrant: {argument.warrant}
        
        OPPOSING POSITION: {opposing_position}
        
        Generate 3 strong counterarguments in JSON format:
        [
            {{
                "claim": "counter-claim 1",
                "grounds": ["grounds"],
                "warrant": "legal principle",
                "backing": ["authorities"],
                "qualifier": "certainty level",
                "rebuttal": null,
                "strength": 0.0-1.0
            }},
            ...
        ]
        """
        
        if self.llm:
            result = await self.llm.generate(
                prompt=prompt,
                task="counterargument",
                model_override=self.model,
                return_json=True
            )
            
            try:
                data = json.loads(result["text"])
                return [LegalArgument(**arg) for arg in data]
            except (json.JSONDecodeError, KeyError):
                return []
        
        return []
    
    # ─── Damages Calculation ───
    
    def calculate_contract_damages(
        self,
        expected_benefit: Decimal,
        actual_performance: Decimal,
        reliance_costs: Decimal = Decimal("0"),
        consequential_damages: Decimal = Decimal("0"),
        mitigation_savings: Decimal = Decimal("0"),
        liquidated_damages: Optional[Decimal] = None
    ) -> DamagesCalculation:
        """
        Calculate contract damages.
        
        Args:
            expected_benefit: Value of expected performance
            actual_performance: Value actually received
            reliance_costs: Out-of-pocket expenses
            consequential_damages: Foreseeable consequential losses
            mitigation_savings: Amounts saved through mitigation
            liquidated_damages: If contract specifies
        
        Returns:
            Damages calculation
        """
        # Use liquidated damages if specified
        if liquidated_damages is not None:
            return DamagesCalculation(
                damage_type=DamageType.LIQUIDATED,
                base_amount=liquidated_damages,
                calculation_notes=["Liquidated damages clause applies"]
            )
        
        # Expectation damages
        expectation = expected_benefit - actual_performance
        
        calc = DamagesCalculation(
            damage_type=DamageType.COMPENSATORY,
            base_amount=expectation,
            adjustments={
                "reliance_costs": reliance_costs,
                "consequential_damages": consequential_damages,
                "mitigation_credit": -mitigation_savings
            },
            calculation_notes=[
                f"Expectation: ${expectation:,.2f}",
                f"Reliance Costs: ${reliance_costs:,.2f}",
                f"Consequential: ${consequential_damages:,.2f}",
                f"Mitigation Credit: -${mitigation_savings:,.2f}"
            ]
        )
        
        calc.calculate_total()
        return calc
    
    def calculate_tort_damages(
        self,
        economic_damages: Decimal,
        non_economic_damages: Decimal = Decimal("0"),
        punitive_multiplier: float = 0,
        state: USState = USState.FEDERAL,
        caps: Optional[Dict[str, Decimal]] = None
    ) -> DamagesCalculation:
        """
        Calculate tort damages with state-specific caps.
        
        Args:
            economic_damages: Quantifiable financial losses
            non_economic_damages: Pain and suffering, etc.
            punitive_multiplier: Multiplier for punitive damages (0 = none)
            state: Jurisdiction for caps
            caps: Optional damage caps
        
        Returns:
            Damages calculation
        """
        # Apply non-economic caps if applicable
        if caps and "non_economic" in caps:
            cap = caps["non_economic"]
            if non_economic_damages > cap:
                non_economic_damages = cap
        
        base = economic_damages + non_economic_damages
        
        calc = DamagesCalculation(
            damage_type=DamageType.COMPENSATORY,
            base_amount=base,
            adjustments={
                "economic": economic_damages,
                "non_economic": non_economic_damages
            },
            calculation_notes=[
                f"Economic Damages: ${economic_damages:,.2f}",
                f"Non-Economic Damages: ${non_economic_damages:,.2f}"
            ],
            jurisdiction_rules={"state": state.value, "caps": caps or {}}
        )
        
        # Add punitive damages if applicable
        if punitive_multiplier > 0:
            punitive = base * Decimal(str(punitive_multiplier))
            
            # Apply punitive caps if any
            if caps and "punitive" in caps:
                punitive = min(punitive, caps["punitive"])
            
            calc.adjustments["punitive_damages"] = punitive
            calc.damage_type = DamageType.PUNITIVE
            calc.calculation_notes.append(
                f"Punitive Damages ({punitive_multiplier}x): ${punitive:,.2f}"
            )
        
        calc.calculate_total()
        return calc
    
    # ─── Statute of Limitations ───
    
    def calculate_sol(
        self,
        state: USState,
        claim_type: str,
        accrual_date: datetime,
        tolling_events: Optional[List[Dict[str, Any]]] = None
    ) -> SOLCalculation:
        """
        Calculate statute of limitations deadline.
        
        Args:
            state: The state
            claim_type: Type of claim
            accrual_date: When cause of action accrued
            tolling_events: Events that toll the SOL
        
        Returns:
            SOL calculation with deadline
        """
        # Get limitation period from state config
        state_config = self.config.state_configs.get(state)
        if not state_config:
            limitation_years = 4  # Default
        else:
            limitation_years = state_config.statute_of_limitations.get(claim_type, 4)
        
        # Calculate tolling days
        tolling_days = 0
        if tolling_events:
            for event in tolling_events:
                tolling_days += event.get("days", 0)
        
        calc = SOLCalculation(
            state=state,
            claim_type=claim_type,
            accrual_date=accrual_date,
            limitation_years=limitation_years,
            tolling_days=tolling_days
        )
        
        calc.calculate()
        return calc
    
    # ─── Legal Syllogism Validation ───
    
    async def validate_syllogism(
        self,
        major_premise: str,  # The rule
        minor_premise: str,  # The facts
        conclusion: str  # The conclusion
    ) -> Dict[str, Any]:
        """
        Validate a legal syllogism for logical soundness.
        
        Args:
            major_premise: The general legal rule
            minor_premise: The specific facts
            conclusion: The proposed conclusion
        
        Returns:
            Validation result with analysis
        """
        prompt = f"""
        Validate this legal syllogism for logical soundness:
        
        MAJOR PREMISE (Rule): {major_premise}
        MINOR PREMISE (Facts): {minor_premise}
        CONCLUSION: {conclusion}
        
        Analyze:
        1. Is the major premise a valid statement of law?
        2. Does the minor premise accurately state the relevant facts?
        3. Does the conclusion logically follow from the premises?
        4. Are there any logical fallacies?
        5. What assumptions are being made?
        
        Return in JSON format:
        {{
            "is_valid": true/false,
            "major_premise_valid": true/false,
            "minor_premise_accurate": true/false,
            "conclusion_follows": true/false,
            "fallacies": ["any logical fallacies"],
            "assumptions": ["underlying assumptions"],
            "strength_score": 0.0-1.0,
            "analysis": "detailed explanation"
        }}
        """
        
        if self.llm:
            result = await self.llm.generate(
                prompt=prompt,
                task="legal_synthesis",
                model_override=self.model,
                return_json=True
            )
            
            try:
                return json.loads(result["text"])
            except json.JSONDecodeError:
                return {
                    "is_valid": False,
                    "analysis": result.get("text", "Unable to validate"),
                    "strength_score": 0.0
                }
        
        return {
            "is_valid": True,
            "analysis": "[Would be validated by LLM]",
            "strength_score": 0.0
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    """Demo the reasoning engine."""
    print("=" * 60)
    print("LEGAL REASONING ENGINE DEMO")
    print("=" * 60)
    
    from configs.config import create_config
    from decimal import Decimal
    
    config = create_config()
    engine = LegalReasoningEngine(config)
    
    # ─── Damages Calculation Demo ───
    print("\n💰 Contract Damages Calculation:")
    damages = engine.calculate_contract_damages(
        expected_benefit=Decimal("100000"),
        actual_performance=Decimal("25000"),
        reliance_costs=Decimal("15000"),
        consequential_damages=Decimal("20000"),
        mitigation_savings=Decimal("5000")
    )
    
    print(f"  Type: {damages.damage_type.value}")
    print(f"  Base (Expectation): ${damages.base_amount:,.2f}")
    for note in damages.calculation_notes:
        print(f"    {note}")
    print(f"  TOTAL: ${damages.total_amount:,.2f}")
    
    # ─── Tort Damages Demo ───
    print("\n💰 Tort Damages Calculation:")
    tort_damages = engine.calculate_tort_damages(
        economic_damages=Decimal("250000"),
        non_economic_damages=Decimal("500000"),
        punitive_multiplier=2.0,
        state=USState.CALIFORNIA,
        caps={"non_economic": Decimal("350000")}  # Example cap
    )
    
    print(f"  Type: {tort_damages.damage_type.value}")
    for note in tort_damages.calculation_notes:
        print(f"    {note}")
    print(f"  TOTAL: ${tort_damages.total_amount:,.2f}")
    
    # ─── SOL Calculation Demo ───
    print("\n⏱️ Statute of Limitations:")
    sol = engine.calculate_sol(
        state=USState.CALIFORNIA,
        claim_type="personal_injury",
        accrual_date=datetime(2024, 6, 15),
        tolling_events=[
            {"reason": "defendant_out_of_state", "days": 30}
        ]
    )
    
    print(f"  State: {sol.state.value}")
    print(f"  Claim Type: {sol.claim_type}")
    print(f"  Accrual Date: {sol.accrual_date.strftime('%Y-%m-%d')}")
    print(f"  Limitation Period: {sol.limitation_years} years")
    print(f"  Tolling: {sol.tolling_days} days")
    print(f"  Deadline: {sol.deadline.strftime('%Y-%m-%d')}")
    print(f"  Status: {'EXPIRED ❌' if sol.is_expired else f'✅ {sol.days_remaining} days remaining'}")
    
    # ─── IRAC Preview ───
    print("\n📋 IRAC Analysis (requires LLM):")
    print("  Would analyze: 'Whether verbal promise constitutes enforceable contract'")
    print("  Framework: Issue → Rule → Application → Conclusion")
    
    # ─── Argument Structure Preview ───
    print("\n🗣️ Argument Construction (requires LLM):")
    print("  Would construct Toulmin-model argument:")
    print("    • Claim: Main assertion")
    print("    • Grounds: Factual basis")
    print("    • Warrant: Legal principle")
    print("    • Backing: Authorities")
    print("    • Qualifier: Certainty level")
    print("    • Rebuttal: Counter-acknowledgment")


if __name__ == "__main__":
    asyncio.run(demo())
