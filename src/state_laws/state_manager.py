"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    STATE LAW MANAGER                                          ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Customizable Infrastructure for State Law Additions:                         ║
║  • Add new state configurations at runtime                                    ║
║  • Load state-specific statute databases                                      ║
║  • State-specific citation formats                                            ║
║  • Statute of limitations by state and domain                                 ║
║  • Court hierarchy management                                                 ║
║  • Special rules per state (anti-SLAPP, etc.)                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import os

from configs.config import (
    LegalAIConfig, 
    StateLawConfig, 
    USState, 
    LegalDomain,
    DEFAULT_STATE_CONFIGS
)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE LAW TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

# Templates for easy state addition
STATE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "common_law": {
        "court_hierarchy": [
            "{state} Supreme Court",
            "{state} Court of Appeals",
            "{state} Superior Court",
            "{state} District Court"
        ],
        "statute_of_limitations": {
            "contract_written": 6,
            "contract_oral": 4,
            "personal_injury": 2,
            "property_damage": 3,
            "fraud": 3,
            "malpractice_legal": 2,
            "malpractice_medical": 2,
        }
    },
    "civil_law": {  # Louisiana-style
        "court_hierarchy": [
            "{state} Supreme Court",
            "{state} Courts of Appeal",
            "{state} District Courts"
        ],
        "statute_of_limitations": {
            "contract_written": 10,
            "contract_oral": 10,
            "personal_injury": 1,
            "property_damage": 1,
            "fraud": 1,
            "delictual_actions": 1,
        },
        "special_rules": {
            "civil_law_system": True,
            "napoleonic_code_influence": True
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# STATE LAW MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class StateLawManager:
    """
    Manager for state-specific legal configurations.
    
    Provides:
    - Easy addition of new state configurations
    - State-specific statute of limitations lookup
    - Citation format generation
    - Court hierarchy queries
    - Special rules checking
    """
    
    def __init__(self, config: LegalAIConfig):
        """Initialize with a base configuration."""
        self.config = config
        self._state_configs: Dict[USState, StateLawConfig] = dict(config.state_configs)
        self._loaded_databases: Dict[USState, bool] = {}
    
    # ─── State Configuration Management ───
    
    def add_state(
        self,
        state: USState,
        state_name: str,
        template: str = "common_law",
        **overrides
    ) -> StateLawConfig:
        """
        Add a new state configuration using a template.
        
        Args:
            state: The USState enum value
            state_name: Full name of the state
            template: Template to use ("common_law" or "civil_law")
            **overrides: Override any template values
        
        Returns:
            The created StateLawConfig
        
        Example:
            manager.add_state(
                USState.FLORIDA,
                "Florida",
                template="common_law",
                statute_of_limitations={"personal_injury": 4}
            )
        """
        if template not in STATE_TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Use 'common_law' or 'civil_law'")
        
        base = STATE_TEMPLATES[template]
        
        # Build court hierarchy with state name
        court_hierarchy = [
            court.format(state=state_name) 
            for court in base["court_hierarchy"]
        ]
        
        # Merge statute of limitations
        sol = dict(base.get("statute_of_limitations", {}))
        if "statute_of_limitations" in overrides:
            sol.update(overrides.pop("statute_of_limitations"))
        
        # Merge special rules
        special_rules = dict(base.get("special_rules", {}))
        if "special_rules" in overrides:
            special_rules.update(overrides.pop("special_rules"))
        
        # Create the config
        state_config = StateLawConfig(
            state=state,
            state_name=state_name,
            court_hierarchy=court_hierarchy,
            statute_of_limitations=sol,
            special_rules=special_rules,
            qdrant_collection=f"{state.value.lower()}_law",
            is_active=True,
            **overrides
        )
        
        self._state_configs[state] = state_config
        return state_config
    
    def add_state_from_json(self, json_path: str) -> StateLawConfig:
        """
        Add a state configuration from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file
        
        Returns:
            The loaded StateLawConfig
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        config = StateLawConfig.from_dict(data)
        self._state_configs[config.state] = config
        return config
    
    def add_state_from_dict(self, data: Dict[str, Any]) -> StateLawConfig:
        """
        Add a state configuration from a dictionary.
        
        Args:
            data: Dictionary with state configuration
        
        Returns:
            The created StateLawConfig
        """
        config = StateLawConfig.from_dict(data)
        self._state_configs[config.state] = config
        return config
    
    def get_state(self, state: USState) -> Optional[StateLawConfig]:
        """Get configuration for a state."""
        return self._state_configs.get(state)
    
    def get_all_states(self) -> List[StateLawConfig]:
        """Get all configured states."""
        return list(self._state_configs.values())
    
    def get_active_states(self) -> List[StateLawConfig]:
        """Get only active (fully configured) states."""
        return [s for s in self._state_configs.values() if s.is_active]
    
    def save_state(self, state: USState, directory: str = "data/state_configs") -> str:
        """
        Save a state configuration to JSON.
        
        Args:
            state: The state to save
            directory: Directory to save to
        
        Returns:
            Path to the saved file
        """
        config = self._state_configs.get(state)
        if not config:
            raise ValueError(f"State not configured: {state}")
        
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / f"{state.value.lower()}_config.json"
        with open(file_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        return str(file_path)
    
    def load_all_from_directory(self, directory: str = "data/state_configs") -> int:
        """
        Load all state configurations from a directory.
        
        Args:
            directory: Directory containing JSON config files
        
        Returns:
            Number of configurations loaded
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0
        
        count = 0
        for json_file in dir_path.glob("*_config.json"):
            try:
                self.add_state_from_json(str(json_file))
                count += 1
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        return count
    
    # ─── Statute of Limitations ───
    
    def get_statute_of_limitations(
        self,
        state: USState,
        claim_type: str
    ) -> Optional[int]:
        """
        Get the statute of limitations for a claim type in a state.
        
        Args:
            state: The state
            claim_type: Type of claim (e.g., "contract_written", "personal_injury")
        
        Returns:
            Number of years, or None if not found
        """
        config = self._state_configs.get(state)
        if not config:
            return None
        return config.statute_of_limitations.get(claim_type)
    
    def calculate_deadline(
        self,
        state: USState,
        claim_type: str,
        accrual_date: datetime,
        tolling_days: int = 0
    ) -> Optional[datetime]:
        """
        Calculate the filing deadline for a claim.
        
        Args:
            state: The state
            claim_type: Type of claim
            accrual_date: Date the cause of action accrued
            tolling_days: Days of tolling to add
        
        Returns:
            Filing deadline, or None if SOL not found
        """
        years = self.get_statute_of_limitations(state, claim_type)
        if years is None:
            return None
        
        # Add years + tolling days
        deadline = accrual_date.replace(year=accrual_date.year + years)
        deadline += timedelta(days=tolling_days)
        
        return deadline
    
    def check_time_barred(
        self,
        state: USState,
        claim_type: str,
        accrual_date: datetime,
        filing_date: Optional[datetime] = None,
        tolling_days: int = 0
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Check if a claim is time-barred.
        
        Args:
            state: The state
            claim_type: Type of claim
            accrual_date: Date cause of action accrued
            filing_date: Date of filing (default: now)
            tolling_days: Days of tolling
        
        Returns:
            Tuple of (is_time_barred, deadline)
        """
        deadline = self.calculate_deadline(state, claim_type, accrual_date, tolling_days)
        if deadline is None:
            return (False, None)  # Can't determine
        
        filing = filing_date or datetime.utcnow()
        return (filing > deadline, deadline)
    
    def compare_sol_across_states(
        self,
        claim_type: str,
        states: Optional[List[USState]] = None
    ) -> Dict[str, int]:
        """
        Compare statute of limitations across states.
        
        Args:
            claim_type: Type of claim
            states: States to compare (default: all active)
        
        Returns:
            Dictionary of state -> years
        """
        states_to_check = states or [s.state for s in self.get_active_states()]
        
        result = {}
        for state in states_to_check:
            years = self.get_statute_of_limitations(state, claim_type)
            if years is not None:
                result[state.value] = years
        
        return dict(sorted(result.items(), key=lambda x: x[1]))
    
    # ─── Citation Formatting ───
    
    def format_statute_citation(
        self,
        state: USState,
        section: str,
        code_type: str = "General"
    ) -> str:
        """
        Format a statute citation for a state.
        
        Args:
            state: The state
            section: Section number
            code_type: Type of code (e.g., "Civil", "Penal", "Business")
        
        Returns:
            Formatted citation string
        """
        config = self._state_configs.get(state)
        if not config:
            return f"{state.value} Code § {section}"
        
        return config.statute_citation_format.format(
            state=config.state_name,
            section=section,
            code_type=code_type
        )
    
    def format_case_citation(
        self,
        state: USState,
        case_name: str,
        volume: str,
        reporter: str,
        page: str,
        court: str,
        year: str,
        series: str = ""
    ) -> str:
        """
        Format a case citation for a state.
        
        Args:
            state: The state
            case_name: Name of the case
            volume: Reporter volume
            reporter: Reporter name
            page: Starting page
            court: Court name
            year: Decision year
            series: Reporter series (e.g., "2d", "3d")
        
        Returns:
            Formatted citation string
        """
        config = self._state_configs.get(state)
        if not config:
            return f"{case_name}, {volume} {reporter} {page} ({court} {year})"
        
        return config.case_citation_format.format(
            case_name=case_name,
            volume=volume,
            reporter=reporter,
            page=page,
            court=court,
            year=year,
            series=series
        )
    
    # ─── Court Hierarchy ───
    
    def get_court_hierarchy(self, state: USState) -> List[str]:
        """Get the court hierarchy for a state (highest to lowest)."""
        config = self._state_configs.get(state)
        if not config:
            return []
        return config.court_hierarchy
    
    def get_highest_court(self, state: USState) -> Optional[str]:
        """Get the highest court in a state."""
        hierarchy = self.get_court_hierarchy(state)
        return hierarchy[0] if hierarchy else None
    
    def is_binding_precedent(
        self,
        state: USState,
        source_court: str,
        target_court: str
    ) -> bool:
        """
        Check if decisions from source_court are binding on target_court.
        
        Returns True if source_court is higher in the hierarchy than target_court.
        """
        hierarchy = self.get_court_hierarchy(state)
        if not hierarchy:
            return False
        
        try:
            source_idx = hierarchy.index(source_court)
            target_idx = hierarchy.index(target_court)
            return source_idx < target_idx  # Lower index = higher court
        except ValueError:
            return False
    
    # ─── Special Rules ───
    
    def has_special_rule(self, state: USState, rule_name: str) -> bool:
        """Check if a state has a specific special rule."""
        config = self._state_configs.get(state)
        if not config:
            return False
        return config.special_rules.get(rule_name, False)
    
    def get_special_rule(self, state: USState, rule_name: str) -> Any:
        """Get a special rule value for a state."""
        config = self._state_configs.get(state)
        if not config:
            return None
        return config.special_rules.get(rule_name)
    
    def find_states_with_rule(self, rule_name: str) -> List[USState]:
        """Find all states that have a specific rule."""
        return [
            config.state 
            for config in self._state_configs.values()
            if config.special_rules.get(rule_name, False)
        ]
    
    # ─── Batch Operations ───
    
    def export_all_configs(self, directory: str = "data/state_configs") -> int:
        """Export all state configurations to JSON files."""
        count = 0
        for state in self._state_configs:
            try:
                self.save_state(state, directory)
                count += 1
            except Exception as e:
                print(f"Failed to save {state}: {e}")
        return count
    
    def generate_state_report(self) -> str:
        """Generate a summary report of all configured states."""
        lines = [
            "=" * 60,
            "STATE LAW CONFIGURATION REPORT",
            "=" * 60,
            ""
        ]
        
        for config in sorted(self._state_configs.values(), key=lambda x: x.state_name):
            status = "✅ Active" if config.is_active else "⚠️ Inactive"
            lines.append(f"📍 {config.state_name} ({config.state.value}) - {status}")
            lines.append(f"   Collection: {config.qdrant_collection or 'Not set'}")
            lines.append(f"   SOL Types: {len(config.statute_of_limitations)}")
            lines.append(f"   Special Rules: {len(config.special_rules)}")
            lines.append("")
        
        lines.append(f"Total: {len(self._state_configs)} states configured")
        lines.append(f"Active: {len(self.get_active_states())} states")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_state_manager(config: Optional[LegalAIConfig] = None) -> StateLawManager:
    """Create a state manager with default configuration."""
    from configs.config import create_config
    cfg = config or create_config()
    return StateLawManager(cfg)


def quick_add_state(
    state_code: str,
    state_name: str,
    **kwargs
) -> StateLawConfig:
    """
    Quickly add a state without full manager setup.
    
    Args:
        state_code: Two-letter state code (e.g., "FL")
        state_name: Full state name
        **kwargs: Additional configuration options
    
    Returns:
        StateLawConfig
    """
    state = USState(state_code)
    manager = create_state_manager()
    return manager.add_state(state, state_name, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo the state law manager."""
    print("=" * 60)
    print("STATE LAW MANAGER DEMO")
    print("=" * 60)
    
    # Create manager
    manager = create_state_manager()
    
    # Show pre-configured states
    print("\n📋 Pre-configured States:")
    print(manager.generate_state_report())
    
    # Add a new state
    print("\n➕ Adding Florida...")
    florida = manager.add_state(
        USState.FLORIDA,
        "Florida",
        template="common_law",
        statute_of_limitations={
            "contract_written": 5,
            "contract_oral": 4,
            "personal_injury": 4,  # Florida has 4 years!
            "medical_malpractice": 2,
        },
        special_rules={
            "no_fault_auto": True,
            "homestead_protection": True,
            "stand_your_ground": True
        }
    )
    print(f"   ✅ Added: {florida.state_name}")
    print(f"   Personal Injury SOL: {florida.statute_of_limitations['personal_injury']} years")
    
    # Compare SOL across states
    print("\n📊 Personal Injury SOL Comparison:")
    comparison = manager.compare_sol_across_states("personal_injury")
    for state, years in comparison.items():
        print(f"   {state}: {years} years")
    
    # Check if claim is time-barred
    print("\n⏱️ Time-Barred Analysis:")
    accrual = datetime(2022, 6, 15)
    is_barred, deadline = manager.check_time_barred(
        USState.CALIFORNIA,
        "personal_injury",
        accrual
    )
    print(f"   State: California")
    print(f"   Accrual Date: {accrual.strftime('%Y-%m-%d')}")
    print(f"   Deadline: {deadline.strftime('%Y-%m-%d') if deadline else 'Unknown'}")
    print(f"   Time-Barred: {'Yes ❌' if is_barred else 'No ✅'}")
    
    # Find states with anti-SLAPP
    print("\n🔍 States with Anti-SLAPP Laws:")
    anti_slapp_states = manager.find_states_with_rule("anti_slapp")
    for state in anti_slapp_states:
        print(f"   - {manager.get_state(state).state_name}")
    
    # Format citations
    print("\n📝 Citation Examples:")
    print(f"   CA Statute: {manager.format_statute_citation(USState.CALIFORNIA, '1542', 'Civil')}")
    print(f"   NY Statute: {manager.format_statute_citation(USState.NEW_YORK, '5-701', 'General Obligations')}")
    
    # Export
    print("\n💾 Exporting configurations...")
    count = manager.export_all_configs("data/state_configs")
    print(f"   Exported {count} state configurations")


if __name__ == "__main__":
    demo()
