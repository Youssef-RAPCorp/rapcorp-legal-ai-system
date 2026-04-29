#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    STATE LAW ADDITION TOOL                                     ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  Customizable Infrastructure for Adding State Laws:                           ║
║  • Interactive CLI for adding new states                                      ║
║  • JSON template generation                                                   ║
║  • Bulk import from directory                                                 ║
║  • State configuration validation                                             ║
║  • Export/Import functionality                                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python add_state.py --interactive              # Interactive wizard
    python add_state.py --template FL Florida      # Generate template
    python add_state.py --import florida.json      # Import from JSON
    python add_state.py --list                     # List all states
    python add_state.py --export-all               # Export all configs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import USState, LegalDomain, StateLawConfig


# ═══════════════════════════════════════════════════════════════════════════════
# STATE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

def get_common_law_template(state_code: str, state_name: str) -> Dict[str, Any]:
    """Generate a common law state template."""
    return {
        "state": state_code,
        "state_name": state_name,
        "statute_db_path": f"data/statutes/{state_code.lower()}/",
        "case_law_db_path": f"data/case_law/{state_code.lower()}/",
        "regulations_db_path": f"data/regulations/{state_code.lower()}/",
        "statute_citation_format": f"{state_name} Code § {{section}}",
        "case_citation_format": "{case_name}, {volume} {reporter} {page} ({court} {year})",
        "court_hierarchy": [
            f"Supreme Court of {state_name}",
            f"{state_name} Court of Appeals",
            f"{state_name} Superior Court",
            f"{state_name} District Court"
        ],
        "statute_of_limitations": {
            "contract_written": 6,
            "contract_oral": 4,
            "personal_injury": 2,
            "property_damage": 3,
            "fraud": 3,
            "malpractice_legal": 2,
            "malpractice_medical": 2,
            "wrongful_death": 2,
            "defamation": 1,
            "employment_discrimination": 1
        },
        "special_rules": {
            "anti_slapp": False,
            "comparative_negligence": True,
            "joint_and_several_liability": True,
            "damage_caps_non_economic": None,
            "damage_caps_punitive": None
        },
        "qdrant_collection": f"{state_code.lower()}_law",
        "is_active": True,
        "_metadata": {
            "created": datetime.utcnow().isoformat(),
            "template": "common_law",
            "version": "1.0"
        }
    }


def get_civil_law_template(state_code: str, state_name: str) -> Dict[str, Any]:
    """Generate a civil law state template (Louisiana-style)."""
    template = get_common_law_template(state_code, state_name)
    template["_metadata"]["template"] = "civil_law"
    template["statute_of_limitations"] = {
        "contract_written": 10,
        "contract_oral": 10,
        "personal_injury": 1,  # Prescriptive period
        "property_damage": 1,
        "fraud": 1,
        "delictual_actions": 1,
        "quasi_contract": 3
    }
    template["special_rules"]["civil_law_system"] = True
    template["special_rules"]["napoleonic_code_influence"] = True
    return template


# ═══════════════════════════════════════════════════════════════════════════════
# STATE DATABASE (Pre-populated hints)
# ═══════════════════════════════════════════════════════════════════════════════

STATE_HINTS = {
    "AL": {"name": "Alabama", "sol_pi": 2, "anti_slapp": False},
    "AK": {"name": "Alaska", "sol_pi": 2, "anti_slapp": False},
    "AZ": {"name": "Arizona", "sol_pi": 2, "anti_slapp": True},
    "AR": {"name": "Arkansas", "sol_pi": 3, "anti_slapp": True},
    "CA": {"name": "California", "sol_pi": 2, "anti_slapp": True},
    "CO": {"name": "Colorado", "sol_pi": 2, "anti_slapp": True},
    "CT": {"name": "Connecticut", "sol_pi": 2, "anti_slapp": True},
    "DE": {"name": "Delaware", "sol_pi": 2, "anti_slapp": True},
    "FL": {"name": "Florida", "sol_pi": 4, "anti_slapp": True},
    "GA": {"name": "Georgia", "sol_pi": 2, "anti_slapp": True},
    "HI": {"name": "Hawaii", "sol_pi": 2, "anti_slapp": True},
    "ID": {"name": "Idaho", "sol_pi": 2, "anti_slapp": False},
    "IL": {"name": "Illinois", "sol_pi": 2, "anti_slapp": True},
    "IN": {"name": "Indiana", "sol_pi": 2, "anti_slapp": True},
    "IA": {"name": "Iowa", "sol_pi": 2, "anti_slapp": False},
    "KS": {"name": "Kansas", "sol_pi": 2, "anti_slapp": True},
    "KY": {"name": "Kentucky", "sol_pi": 1, "anti_slapp": False},
    "LA": {"name": "Louisiana", "sol_pi": 1, "anti_slapp": True, "civil_law": True},
    "ME": {"name": "Maine", "sol_pi": 6, "anti_slapp": True},
    "MD": {"name": "Maryland", "sol_pi": 3, "anti_slapp": True},
    "MA": {"name": "Massachusetts", "sol_pi": 3, "anti_slapp": True},
    "MI": {"name": "Michigan", "sol_pi": 3, "anti_slapp": False},
    "MN": {"name": "Minnesota", "sol_pi": 2, "anti_slapp": True},
    "MS": {"name": "Mississippi", "sol_pi": 3, "anti_slapp": False},
    "MO": {"name": "Missouri", "sol_pi": 5, "anti_slapp": True},
    "MT": {"name": "Montana", "sol_pi": 3, "anti_slapp": False},
    "NE": {"name": "Nebraska", "sol_pi": 4, "anti_slapp": True},
    "NV": {"name": "Nevada", "sol_pi": 2, "anti_slapp": True},
    "NH": {"name": "New Hampshire", "sol_pi": 3, "anti_slapp": False},
    "NJ": {"name": "New Jersey", "sol_pi": 2, "anti_slapp": False},
    "NM": {"name": "New Mexico", "sol_pi": 3, "anti_slapp": True},
    "NY": {"name": "New York", "sol_pi": 3, "anti_slapp": False},
    "NC": {"name": "North Carolina", "sol_pi": 3, "anti_slapp": False},
    "ND": {"name": "North Dakota", "sol_pi": 6, "anti_slapp": False},
    "OH": {"name": "Ohio", "sol_pi": 2, "anti_slapp": False},
    "OK": {"name": "Oklahoma", "sol_pi": 2, "anti_slapp": True},
    "OR": {"name": "Oregon", "sol_pi": 2, "anti_slapp": True},
    "PA": {"name": "Pennsylvania", "sol_pi": 2, "anti_slapp": True},
    "RI": {"name": "Rhode Island", "sol_pi": 3, "anti_slapp": True},
    "SC": {"name": "South Carolina", "sol_pi": 3, "anti_slapp": False},
    "SD": {"name": "South Dakota", "sol_pi": 3, "anti_slapp": False},
    "TN": {"name": "Tennessee", "sol_pi": 1, "anti_slapp": True},
    "TX": {"name": "Texas", "sol_pi": 2, "anti_slapp": True},
    "UT": {"name": "Utah", "sol_pi": 4, "anti_slapp": True},
    "VT": {"name": "Vermont", "sol_pi": 3, "anti_slapp": True},
    "VA": {"name": "Virginia", "sol_pi": 2, "anti_slapp": False},
    "WA": {"name": "Washington", "sol_pi": 3, "anti_slapp": True},
    "WV": {"name": "West Virginia", "sol_pi": 2, "anti_slapp": False},
    "WI": {"name": "Wisconsin", "sol_pi": 3, "anti_slapp": False},
    "WY": {"name": "Wyoming", "sol_pi": 4, "anti_slapp": False},
    "DC": {"name": "District of Columbia", "sol_pi": 3, "anti_slapp": True},
}


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE WIZARD
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_add_state() -> Dict[str, Any]:
    """Interactive wizard for adding a new state."""
    print("\n" + "=" * 60)
    print("       STATE LAW ADDITION WIZARD")
    print("=" * 60)
    
    # Get state code
    print("\nAvailable state codes:")
    codes = list(STATE_HINTS.keys())
    for i, code in enumerate(codes):
        hint = STATE_HINTS[code]
        print(f"  {code}: {hint['name']}", end="")
        if (i + 1) % 5 == 0:
            print()
    print("\n")
    
    state_code = input("Enter state code (e.g., FL): ").upper().strip()
    
    if state_code in STATE_HINTS:
        hint = STATE_HINTS[state_code]
        state_name = hint["name"]
        print(f"  → Auto-detected: {state_name}")
        
        # Use civil law template for Louisiana
        if hint.get("civil_law"):
            template = get_civil_law_template(state_code, state_name)
            print("  → Using Civil Law template (Louisiana-style)")
        else:
            template = get_common_law_template(state_code, state_name)
        
        # Apply hints
        template["statute_of_limitations"]["personal_injury"] = hint["sol_pi"]
        template["special_rules"]["anti_slapp"] = hint["anti_slapp"]
    else:
        state_name = input("Enter full state name: ").strip()
        template = get_common_law_template(state_code, state_name)
    
    # Customize SOL
    print("\n📋 STATUTE OF LIMITATIONS (in years)")
    print("   Press Enter to keep default values")
    
    sol_types = [
        ("contract_written", "Written Contract"),
        ("contract_oral", "Oral Contract"),
        ("personal_injury", "Personal Injury"),
        ("property_damage", "Property Damage"),
        ("fraud", "Fraud"),
        ("malpractice_legal", "Legal Malpractice"),
        ("malpractice_medical", "Medical Malpractice"),
    ]
    
    for key, name in sol_types:
        current = template["statute_of_limitations"].get(key, 2)
        user_input = input(f"   {name} [{current}]: ").strip()
        if user_input:
            try:
                template["statute_of_limitations"][key] = int(user_input)
            except ValueError:
                print(f"     Invalid, keeping {current}")
    
    # Special rules
    print("\n⚖️ SPECIAL RULES (y/n)")
    
    rules = [
        ("anti_slapp", "Anti-SLAPP Law"),
        ("comparative_negligence", "Comparative Negligence"),
        ("joint_and_several_liability", "Joint and Several Liability"),
    ]
    
    for key, name in rules:
        current = template["special_rules"].get(key, False)
        default = "y" if current else "n"
        user_input = input(f"   {name} [{default}]: ").strip().lower()
        if user_input in ("y", "yes"):
            template["special_rules"][key] = True
        elif user_input in ("n", "no"):
            template["special_rules"][key] = False
    
    # Damage caps
    print("\n💰 DAMAGE CAPS (leave blank for none)")
    
    non_econ = input("   Non-Economic Damages Cap ($): ").strip()
    if non_econ:
        try:
            template["special_rules"]["damage_caps_non_economic"] = int(non_econ)
        except ValueError:
            pass
    
    punitive = input("   Punitive Damages Cap ($): ").strip()
    if punitive:
        try:
            template["special_rules"]["damage_caps_punitive"] = int(punitive)
        except ValueError:
            pass
    
    return template


def save_state_config(config: Dict[str, Any], output_dir: str = "data/state_configs") -> str:
    """Save a state configuration to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    state_code = config["state"].lower()
    file_path = output_path / f"{state_code}_config.json"
    
    with open(file_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return str(file_path)


def load_state_config(json_path: str) -> Dict[str, Any]:
    """Load a state configuration from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def list_all_states(config_dir: str = "data/state_configs") -> List[Dict[str, Any]]:
    """List all configured states."""
    states = []
    config_path = Path(config_dir)
    
    if config_path.exists():
        for json_file in config_path.glob("*_config.json"):
            try:
                with open(json_file, "r") as f:
                    config = json.load(f)
                    states.append({
                        "file": str(json_file),
                        "state": config.get("state"),
                        "name": config.get("state_name"),
                        "active": config.get("is_active", False)
                    })
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
    
    return states


def validate_state_config(config: Dict[str, Any]) -> List[str]:
    """Validate a state configuration and return any errors."""
    errors = []
    
    required_fields = ["state", "state_name", "statute_of_limitations", "court_hierarchy"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate state code
    if "state" in config:
        try:
            USState(config["state"])
        except ValueError:
            errors.append(f"Invalid state code: {config['state']}")
    
    # Validate SOL values
    if "statute_of_limitations" in config:
        for key, value in config["statute_of_limitations"].items():
            if not isinstance(value, (int, float)) or value < 0:
                errors.append(f"Invalid SOL value for {key}: {value}")
    
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# CLI MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAPCorp Legal AI - State Law Addition Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_state.py --interactive              # Interactive wizard
  python add_state.py --template FL Florida      # Generate FL template
  python add_state.py --import florida.json      # Import from JSON
  python add_state.py --list                     # List configured states
  python add_state.py --export-all               # Export all to JSON
        """
    )
    
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Run interactive state addition wizard")
    parser.add_argument("--template", "-t", nargs=2, metavar=("CODE", "NAME"),
                        help="Generate a template for a state")
    parser.add_argument("--import", "-m", dest="import_file", metavar="FILE",
                        help="Import state config from JSON file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all configured states")
    parser.add_argument("--validate", "-v", metavar="FILE",
                        help="Validate a state config file")
    parser.add_argument("--export-all", "-e", action="store_true",
                        help="Export all state configs")
    parser.add_argument("--output", "-o", default="data/state_configs",
                        help="Output directory for configs")
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        config = interactive_add_state()
        
        print("\n" + "=" * 60)
        print("GENERATED CONFIGURATION:")
        print("=" * 60)
        print(json.dumps(config, indent=2))
        
        save = input("\nSave this configuration? (y/n): ").strip().lower()
        if save in ("y", "yes"):
            path = save_state_config(config, args.output)
            print(f"✅ Saved to: {path}")
        return
    
    # Generate template
    if args.template:
        code, name = args.template
        code = code.upper()
        
        if code == "LA":
            config = get_civil_law_template(code, name)
        else:
            config = get_common_law_template(code, name)
        
        path = save_state_config(config, args.output)
        print(f"✅ Template saved to: {path}")
        print("\nEdit the JSON file to customize, then use --import to validate.")
        return
    
    # Import from file
    if args.import_file:
        config = load_state_config(args.import_file)
        errors = validate_state_config(config)
        
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"   - {error}")
            return
        
        path = save_state_config(config, args.output)
        print(f"✅ Imported and saved to: {path}")
        return
    
    # List states
    if args.list:
        states = list_all_states(args.output)
        
        print("\n" + "=" * 60)
        print("CONFIGURED STATES")
        print("=" * 60)
        
        if not states:
            print("No states configured yet.")
            print("Use --interactive or --template to add states.")
        else:
            for state in states:
                status = "✅" if state["active"] else "⚠️"
                print(f"  {status} {state['state']}: {state['name']}")
        return
    
    # Validate
    if args.validate:
        config = load_state_config(args.validate)
        errors = validate_state_config(config)
        
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"   - {error}")
        else:
            print("✅ Configuration is valid!")
        return
    
    # Export all
    if args.export_all:
        states = list_all_states(args.output)
        print(f"Found {len(states)} state configurations in {args.output}")
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
