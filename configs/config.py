"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RAPCorp LEGAL AI SYSTEM - CONFIGURATION                    ║
║              Renaissance of American Physics and Astronomy (RAPCorp)          ║
║                                                                               ║
║  Gemini Models:                                                               ║
║  • gemini-flash-latest      - Fast retrieval & simple tasks                   ║
║  • gemini-pro-latest        - Deep reasoning & analysis                       ║
║  • gemini-flash-lite-latest - Ultra-fast, cost-optimized                      ║
║                                                                               ║
║  State Law Customization:                                                     ║
║  • Modular state configurations                                               ║
║  • Jurisdiction-specific rules                                                ║
║  • State statute databases                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any
from pathlib import Path
import os
import json


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiModel(Enum):
    """
    Specific Gemini models for Legal AI tasks.
    
    Model Selection Strategy:
    - FLASH_LITE: Simple queries, embeddings prep, quick lookups (~$0.01/query)
    - FLASH: Standard retrieval, summarization, drafting (~$0.02/query)
    - PRO: Complex reasoning, IRAC analysis, argument construction (~$0.05/query)
    """
    FLASH_LITE = "gemini-flash-lite-latest"   # Ultra-fast, cheapest
    FLASH = "gemini-flash-latest"              # Balanced speed/quality
    PRO = "gemini-pro-latest"                  # Maximum reasoning power


@dataclass
class ModelConfig:
    """Configuration for a specific Gemini model."""
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    use_cases: List[str]
    cost_per_1k_input: float
    cost_per_1k_output: float


# Default model configurations
MODEL_CONFIGS: Dict[GeminiModel, ModelConfig] = {
    GeminiModel.FLASH_LITE: ModelConfig(
        model_id="gemini-flash-lite-latest",
        max_tokens=2048,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        use_cases=[
            "document_classification",
            "simple_extraction",
            "yes_no_questions",
            "keyword_extraction",
            "quick_summaries"
        ],
        cost_per_1k_input=0.00001,
        cost_per_1k_output=0.00002
    ),
    GeminiModel.FLASH: ModelConfig(
        model_id="gemini-flash-latest",
        max_tokens=8192,
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        use_cases=[
            "document_retrieval",
            "case_summarization",
            "contract_parsing",
            "citation_extraction",
            "standard_drafting",
            "swarm_agents"
        ],
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.0001
    ),
    GeminiModel.PRO: ModelConfig(
        model_id="gemini-pro-latest",
        max_tokens=32768,
        temperature=0.5,
        top_p=0.95,
        top_k=64,
        use_cases=[
            "irac_analysis",
            "creac_analysis",
            "complex_reasoning",
            "argument_construction",
            "counterargument_generation",
            "legal_synthesis",
            "damages_calculation",
            "risk_assessment",
            "final_review"
        ],
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.001
    )
}


# ═══════════════════════════════════════════════════════════════════════════════
# STATE LAW CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class USState(Enum):
    """All 50 US States + DC and Territories."""
    ALABAMA = "AL"
    ALASKA = "AK"
    ARIZONA = "AZ"
    ARKANSAS = "AR"
    CALIFORNIA = "CA"
    COLORADO = "CO"
    CONNECTICUT = "CT"
    DELAWARE = "DE"
    FLORIDA = "FL"
    GEORGIA = "GA"
    HAWAII = "HI"
    IDAHO = "ID"
    ILLINOIS = "IL"
    INDIANA = "IN"
    IOWA = "IA"
    KANSAS = "KS"
    KENTUCKY = "KY"
    LOUISIANA = "LA"
    MAINE = "ME"
    MARYLAND = "MD"
    MASSACHUSETTS = "MA"
    MICHIGAN = "MI"
    MINNESOTA = "MN"
    MISSISSIPPI = "MS"
    MISSOURI = "MO"
    MONTANA = "MT"
    NEBRASKA = "NE"
    NEVADA = "NV"
    NEW_HAMPSHIRE = "NH"
    NEW_JERSEY = "NJ"
    NEW_MEXICO = "NM"
    NEW_YORK = "NY"
    NORTH_CAROLINA = "NC"
    NORTH_DAKOTA = "ND"
    OHIO = "OH"
    OKLAHOMA = "OK"
    OREGON = "OR"
    PENNSYLVANIA = "PA"
    RHODE_ISLAND = "RI"
    SOUTH_CAROLINA = "SC"
    SOUTH_DAKOTA = "SD"
    TENNESSEE = "TN"
    TEXAS = "TX"
    UTAH = "UT"
    VERMONT = "VT"
    VIRGINIA = "VA"
    WASHINGTON = "WA"
    WEST_VIRGINIA = "WV"
    WISCONSIN = "WI"
    WYOMING = "WY"
    DISTRICT_OF_COLUMBIA = "DC"
    PUERTO_RICO = "PR"
    GUAM = "GU"
    VIRGIN_ISLANDS = "VI"
    FEDERAL = "FED"  # Federal jurisdiction


class LegalDomain(Enum):
    """Legal practice areas."""
    CONTRACT = "contract"
    TORT = "tort"
    CRIMINAL = "criminal"
    FAMILY = "family"
    EMPLOYMENT = "employment"
    REAL_ESTATE = "real_estate"
    INTELLECTUAL_PROPERTY = "ip"
    CORPORATE = "corporate"
    BANKRUPTCY = "bankruptcy"
    IMMIGRATION = "immigration"
    TAX = "tax"
    ENVIRONMENTAL = "environmental"
    HEALTHCARE = "healthcare"
    CIVIL_RIGHTS = "civil_rights"
    ADMINISTRATIVE = "administrative"


@dataclass
class StateLawConfig:
    """
    Configuration for a specific state's legal system.
    
    This enables customizable infrastructure for state law additions.
    Each state can have its own:
    - Statute database paths
    - Citation formats
    - Court hierarchy
    - Statute of limitations rules
    - Specific legal requirements
    """
    state: USState
    state_name: str
    
    # Database paths
    statute_db_path: Optional[str] = None
    case_law_db_path: Optional[str] = None
    regulations_db_path: Optional[str] = None
    
    # Citation format (e.g., "Cal. Civ. Code § {section}")
    statute_citation_format: str = "{state} Code § {section}"
    case_citation_format: str = "{case_name}, {volume} {reporter} {page} ({court} {year})"
    
    # Court hierarchy (highest to lowest)
    court_hierarchy: List[str] = field(default_factory=list)
    
    # Statute of limitations (domain -> years)
    statute_of_limitations: Dict[str, int] = field(default_factory=dict)
    
    # State-specific rules
    special_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Vector collection name for this state
    qdrant_collection: Optional[str] = None
    
    # Is this state fully configured?
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "state": self.state.value,
            "state_name": self.state_name,
            "statute_db_path": self.statute_db_path,
            "case_law_db_path": self.case_law_db_path,
            "regulations_db_path": self.regulations_db_path,
            "statute_citation_format": self.statute_citation_format,
            "case_citation_format": self.case_citation_format,
            "court_hierarchy": self.court_hierarchy,
            "statute_of_limitations": self.statute_of_limitations,
            "special_rules": self.special_rules,
            "qdrant_collection": self.qdrant_collection,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateLawConfig":
        """Create from dictionary."""
        return cls(
            state=USState(data["state"]),
            state_name=data["state_name"],
            statute_db_path=data.get("statute_db_path"),
            case_law_db_path=data.get("case_law_db_path"),
            regulations_db_path=data.get("regulations_db_path"),
            statute_citation_format=data.get("statute_citation_format", "{state} Code § {section}"),
            case_citation_format=data.get("case_citation_format", "{case_name}, {volume} {reporter} {page} ({court} {year})"),
            court_hierarchy=data.get("court_hierarchy", []),
            statute_of_limitations=data.get("statute_of_limitations", {}),
            special_rules=data.get("special_rules", {}),
            qdrant_collection=data.get("qdrant_collection"),
            is_active=data.get("is_active", False)
        )


# Pre-configured state examples (can be extended)
DEFAULT_STATE_CONFIGS: Dict[USState, StateLawConfig] = {
    USState.CALIFORNIA: StateLawConfig(
        state=USState.CALIFORNIA,
        state_name="California",
        statute_citation_format="Cal. {code_type} Code § {section}",
        case_citation_format="{case_name}, {volume} Cal.{series} {page} ({year})",
        court_hierarchy=[
            "Supreme Court of California",
            "California Courts of Appeal",
            "California Superior Courts"
        ],
        statute_of_limitations={
            "contract_written": 4,
            "contract_oral": 2,
            "personal_injury": 2,
            "property_damage": 3,
            "fraud": 3,
            "malpractice_legal": 1,
            "malpractice_medical": 1,
            "employment_discrimination": 1,
        },
        special_rules={
            "anti_slapp": True,
            "proposition_65": True,
            "ccpa_privacy": True,
            "ab5_worker_classification": True
        },
        qdrant_collection="california_law",
        is_active=True
    ),
    USState.NEW_YORK: StateLawConfig(
        state=USState.NEW_YORK,
        state_name="New York",
        statute_citation_format="N.Y. {code_type} Law § {section}",
        case_citation_format="{case_name}, {volume} N.Y.{series} {page} ({year})",
        court_hierarchy=[
            "New York Court of Appeals",
            "New York Appellate Division",
            "New York Supreme Court",
            "New York County Courts"
        ],
        statute_of_limitations={
            "contract_written": 6,
            "contract_oral": 6,
            "personal_injury": 3,
            "property_damage": 3,
            "fraud": 6,
            "malpractice_legal": 3,
            "malpractice_medical": 2.5,
        },
        special_rules={
            "martin_act_securities": True,
            "rent_stabilization": True
        },
        qdrant_collection="new_york_law",
        is_active=True
    ),
    USState.TEXAS: StateLawConfig(
        state=USState.TEXAS,
        state_name="Texas",
        statute_citation_format="Tex. {code_type} Code Ann. § {section}",
        case_citation_format="{case_name}, {volume} S.W.{series} {page} (Tex. {court} {year})",
        court_hierarchy=[
            "Supreme Court of Texas",
            "Texas Court of Criminal Appeals",
            "Texas Courts of Appeals",
            "Texas District Courts"
        ],
        statute_of_limitations={
            "contract_written": 4,
            "contract_oral": 4,
            "personal_injury": 2,
            "property_damage": 2,
            "fraud": 4,
            "malpractice_legal": 2,
            "malpractice_medical": 2,
        },
        special_rules={
            "texas_citizen_participation_act": True,  # Anti-SLAPP
            "chapter_74_medical_malpractice": True,
            "loser_pays_limited": True
        },
        qdrant_collection="texas_law",
        is_active=True
    ),
    USState.FEDERAL: StateLawConfig(
        state=USState.FEDERAL,
        state_name="Federal",
        statute_citation_format="{title} U.S.C. § {section}",
        case_citation_format="{case_name}, {volume} {reporter} {page} ({court} {year})",
        court_hierarchy=[
            "Supreme Court of the United States",
            "United States Courts of Appeals",
            "United States District Courts",
            "United States Bankruptcy Courts"
        ],
        statute_of_limitations={
            "federal_civil_rights": 2,  # Borrows from state
            "federal_antitrust": 4,
            "federal_securities": 2,
            "federal_tax": 3,
            "federal_patent": 6,
            "federal_trademark": 3,
            "federal_copyright": 3,
        },
        special_rules={
            "diversity_jurisdiction_amount": 75000,
            "federal_question_jurisdiction": True,
            "erie_doctrine": True
        },
        qdrant_collection="federal_law",
        is_active=True
    ),
    USState.NEBRASKA: StateLawConfig(  # covers both Douglas County and Sarpy County
        state=USState.NEBRASKA,
        state_name="Nebraska",
        statute_citation_format="Neb. Rev. Stat. § {section}",
        case_citation_format="{case_name}, {volume} Neb. {page} ({year})",
        court_hierarchy=[
            "Nebraska Supreme Court",
            "Nebraska Court of Appeals",
            "Nebraska District Courts",
            "Nebraska County Courts"
        ],
        statute_of_limitations={
            "contract_written": 5,
            "contract_oral": 4,
            "personal_injury": 4,
            "property_damage": 4,
            "fraud": 4,
            "malpractice_legal": 2,
            "malpractice_medical": 2,
            "civil_rights": 4,
        },
        special_rules={
            # Mental Health Commitment Act — Neb. Rev. Stat. §§ 71-901 to 71-962
            "mental_health_commitment_act": True,
            # Standard for involuntary commitment: mentally ill AND dangerous to self/others
            # Neb. Rev. Stat. § 71-919
            "commitment_requires_dangerousness_finding": True,
            # Outpatient commitment termination: Neb. Rev. Stat. § 71-948
            "outpatient_commitment_termination_statute": "Neb. Rev. Stat. § 71-948",
            # Motion to Discharge filed in existing docket
            "discharge_motion_in_existing_docket": True,
            # Board of Mental Health evaluates commitment — conflict of interest rules apply
            "board_of_mental_health_conflict_of_interest_rule": True,
            # Douglas County District Court is the relevant court for Douglas County
            "primary_mental_health_court": "District Court of Douglas County, Nebraska",
            # Harassment protection order standard — Neb. Rev. Stat. § 28-311.09
            # Pattern of conduct that seriously terrifies/intimidates AND serves no legitimate purpose
            "harassment_protection_order_statute": "Neb. Rev. Stat. § 28-311.09",
            # Sarpy County District Court handles cases arising in Sarpy County
            "sarpy_county_court": "District Court of Sarpy County, Nebraska",
        },
        qdrant_collection="nebraska_law",
        is_active=True
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegalAIConfig:
    """
    Master configuration for the Legal AI System.
    
    Supports:
    - Multiple Gemini models for different tasks
    - Customizable state law infrastructure
    - Cost tracking and budgets
    - Vector database settings
    - MCP server integrations
    """
    
    # ─── API Keys (from environment) ───
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    courtlistener_api_key: str = field(default_factory=lambda: os.getenv("COURTLISTENER_API_KEY", ""))
    westlaw_api_key: str = field(default_factory=lambda: os.getenv("WESTLAW_API_KEY", ""))
    lexisnexis_api_key: str = field(default_factory=lambda: os.getenv("LEXISNEXIS_API_KEY", ""))
    
    # ─── Gemini Model Selection ───
    model_flash_lite: str = GeminiModel.FLASH_LITE.value
    model_flash: str = GeminiModel.FLASH.value
    model_pro: str = GeminiModel.PRO.value
    
    # Task-to-model mapping
    task_model_mapping: Dict[str, str] = field(default_factory=lambda: {
        # Flash Lite tasks (cheapest, fastest)
        "classification": GeminiModel.FLASH_LITE.value,
        "keyword_extraction": GeminiModel.FLASH_LITE.value,
        "yes_no_question": GeminiModel.FLASH_LITE.value,
        "simple_lookup": GeminiModel.FLASH_LITE.value,
        
        # Flash tasks (balanced)
        "retrieval": GeminiModel.FLASH.value,
        "summarization": GeminiModel.FLASH.value,
        "citation_extraction": GeminiModel.FLASH.value,
        "contract_parsing": GeminiModel.FLASH.value,
        "swarm_agent": GeminiModel.FLASH.value,
        "draft_generation": GeminiModel.FLASH.value,
        
        # Pro tasks (maximum reasoning)
        "irac_analysis": GeminiModel.PRO.value,
        "creac_analysis": GeminiModel.PRO.value,
        "argument_construction": GeminiModel.PRO.value,
        "counterargument": GeminiModel.PRO.value,
        "legal_synthesis": GeminiModel.PRO.value,
        "risk_assessment": GeminiModel.PRO.value,
        "damages_calculation": GeminiModel.PRO.value,
        "final_review": GeminiModel.PRO.value,
    })
    
    # ─── State Law Configuration ───
    enabled_states: List[USState] = field(default_factory=lambda: [
        USState.CALIFORNIA,
        USState.NEW_YORK,
        USState.TEXAS,
        USState.FEDERAL,
        USState.NEBRASKA,
    ])
    state_configs: Dict[USState, StateLawConfig] = field(default_factory=lambda: DEFAULT_STATE_CONFIGS.copy())
    state_config_dir: str = "data/state_configs"
    
    # ─── Vector Database ───
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    embedding_model: str = "all-MiniLM-L6-v2"  # Local, FREE
    embedding_dimension: int = 384
    
    # Collection names
    collections: Dict[str, str] = field(default_factory=lambda: {
        "case_law": "legal_case_law",
        "statutes": "legal_statutes",
        "regulations": "legal_regulations",
        "contracts": "legal_contracts",
        "briefs": "legal_briefs",
        "legal_journals": "legal_journals"
    })
    
    # ─── Cost Management ───
    daily_budget_usd: float = 5.00
    monthly_budget_usd: float = 100.00
    cost_alert_threshold: float = 0.80  # Alert at 80% of budget
    track_costs: bool = True
    
    # ─── RAG Pipeline ───
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    min_similarity_score: float = 0.7
    use_hybrid_search: bool = True
    
    # ─── Reasoning Engine ───
    enable_irac: bool = True
    enable_creac: bool = True
    enable_counterarguments: bool = True
    reasoning_chain_max_steps: int = 10
    
    # ─── Control Plane ───
    require_human_review: bool = True
    human_review_threshold: float = 0.8  # Confidence below this triggers review
    enable_audit_trail: bool = True
    upl_safeguards: bool = True  # Unauthorized Practice of Law protections
    
    # ─── Memory Bank ───
    enable_case_memory: bool = True
    memory_retention_years: int = 5
    
    # ─── Paths ───
    base_dir: str = field(default_factory=lambda: str(Path(__file__).parent.parent))
    data_dir: str = "data"
    logs_dir: str = "logs"
    templates_dir: str = "templates"
    
    def get_model_for_task(self, task: str) -> str:
        """Get the appropriate Gemini model for a task."""
        return self.task_model_mapping.get(task, self.model_flash)
    
    def get_state_config(self, state: USState) -> Optional[StateLawConfig]:
        """Get configuration for a specific state."""
        return self.state_configs.get(state)
    
    def add_state_config(self, config: StateLawConfig) -> None:
        """Add or update a state configuration."""
        self.state_configs[config.state] = config
        if config.state not in self.enabled_states:
            self.enabled_states.append(config.state)
    
    def load_state_configs_from_dir(self, dir_path: Optional[str] = None) -> None:
        """Load state configurations from JSON files in a directory."""
        config_dir = Path(dir_path or self.state_config_dir)
        if not config_dir.exists():
            return
        
        for json_file in config_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    config = StateLawConfig.from_dict(data)
                    self.add_state_config(config)
            except Exception as e:
                print(f"Warning: Failed to load state config from {json_file}: {e}")
    
    def save_state_config(self, state: USState, dir_path: Optional[str] = None) -> None:
        """Save a state configuration to a JSON file."""
        config = self.state_configs.get(state)
        if not config:
            raise ValueError(f"No configuration found for state: {state}")
        
        config_dir = Path(dir_path or self.state_config_dir)
        config_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = config_dir / f"{state.value.lower()}_config.json"
        with open(file_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "models": {
                "flash_lite": self.model_flash_lite,
                "flash": self.model_flash,
                "pro": self.model_pro,
                "task_mapping": self.task_model_mapping
            },
            "enabled_states": [s.value for s in self.enabled_states],
            "vector_db": {
                "host": self.qdrant_host,
                "port": self.qdrant_port,
                "embedding_model": self.embedding_model,
                "collections": self.collections
            },
            "cost_management": {
                "daily_budget_usd": self.daily_budget_usd,
                "monthly_budget_usd": self.monthly_budget_usd,
                "alert_threshold": self.cost_alert_threshold
            },
            "rag_pipeline": {
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_k": self.rerank_top_k,
                "min_similarity_score": self.min_similarity_score,
                "use_hybrid_search": self.use_hybrid_search
            },
            "reasoning_engine": {
                "enable_irac": self.enable_irac,
                "enable_creac": self.enable_creac,
                "enable_counterarguments": self.enable_counterarguments,
                "max_steps": self.reasoning_chain_max_steps
            },
            "control_plane": {
                "require_human_review": self.require_human_review,
                "human_review_threshold": self.human_review_threshold,
                "enable_audit_trail": self.enable_audit_trail,
                "upl_safeguards": self.upl_safeguards
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_config(
    env_file: Optional[str] = None,
    custom_states: Optional[List[StateLawConfig]] = None,
    **overrides
) -> LegalAIConfig:
    """
    Factory function to create a LegalAIConfig instance.
    
    Args:
        env_file: Path to .env file (optional)
        custom_states: List of custom state configurations to add
        **overrides: Any config parameters to override
    
    Returns:
        Configured LegalAIConfig instance
    """
    # Load environment variables — always attempt to load .env from project root
    from dotenv import load_dotenv
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        # Auto-detect .env next to configs/ or in the project root
        default_env = Path(__file__).parent.parent / ".env"
        if default_env.exists():
            load_dotenv(str(default_env), override=True)

    # Create base config
    config = LegalAIConfig(**overrides)
    
    # Load state configs from directory
    config.load_state_configs_from_dir()
    
    # Add any custom states
    if custom_states:
        for state_config in custom_states:
            config.add_state_config(state_config)
    
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT TEMPLATE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_env_template() -> str:
    """Generate a .env template file content."""
    return """# ═══════════════════════════════════════════════════════════════════════════════
# RAPCorp LEGAL AI SYSTEM - ENVIRONMENT CONFIGURATION
# Renaissance of American Physics and Astronomy (RAPCorp)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── REQUIRED: Google API ───
GOOGLE_API_KEY=your_google_api_key_here

# ─── RECOMMENDED: Free Legal Database ───
COURTLISTENER_API_KEY=your_courtlistener_key_here

# ─── OPTIONAL: Commercial Legal Databases ───
WESTLAW_API_KEY=
LEXISNEXIS_API_KEY=
PACER_USERNAME=
PACER_PASSWORD=

# ─── OPTIONAL: Vector Database ───
QDRANT_API_KEY=
QDRANT_HOST=localhost
QDRANT_PORT=6333

# ─── OPTIONAL: Cost Management ───
DAILY_BUDGET_USD=5.00
MONTHLY_BUDGET_USD=100.00

# ─── OPTIONAL: Logging ───
LOG_LEVEL=INFO
"""


if __name__ == "__main__":
    # Demo: Create and display configuration
    config = create_config()
    
    print("=" * 60)
    print("RAPCorp LEGAL AI SYSTEM - CONFIGURATION")
    print("=" * 60)
    
    print("\n📊 Gemini Models:")
    print(f"  • Flash Lite: {config.model_flash_lite}")
    print(f"  • Flash:      {config.model_flash}")
    print(f"  • Pro:        {config.model_pro}")
    
    print("\n🗺️ Enabled States:")
    for state in config.enabled_states:
        state_config = config.get_state_config(state)
        status = "✅" if state_config and state_config.is_active else "⚠️"
        print(f"  {status} {state.value}: {state_config.state_name if state_config else 'Not configured'}")
    
    print("\n💰 Cost Management:")
    print(f"  • Daily Budget:   ${config.daily_budget_usd:.2f}")
    print(f"  • Monthly Budget: ${config.monthly_budget_usd:.2f}")
    
    print("\n" + "=" * 60)
    print("Environment Template (.env.template):")
    print("=" * 60)
    print(generate_env_template())
