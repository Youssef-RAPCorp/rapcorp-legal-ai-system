# RAPCorp Legal AI System

> **Renaissance of American Physics and Astronomy (RAPCorp)** - Legal AI Infrastructure

A comprehensive Legal AI platform powered by Google's Gemini models with customizable state law infrastructure.

## 🚀 Features

### Gemini Models (Your Specific Request!)
- **`gemini-flash-lite-latest`** - Ultra-fast, cost-optimized (~$0.01/query)
- **`gemini-flash-latest`** - Balanced speed/quality (~$0.02/query)
- **`gemini-pro-latest`** - Maximum reasoning power (~$0.05/query)

### Customizable State Law Infrastructure (Your Specific Request!)
- Add new states interactively or via JSON
- State-specific statute of limitations
- Custom citation formats per state
- Court hierarchy management
- Special rules (anti-SLAPP, damage caps, etc.)

### Core Components
- **15-Agent Knowledge Swarm** - Parallel legal research
- **IRAC/CREAC Reasoning Engine** - Structured legal analysis
- **3-Stage RAG Pipeline** - Retrieval → Analysis → Synthesis
- **Control Plane** - Audit trail, UPL safeguards

---

## 📦 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rapcorp-legal-ai-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env
# Edit .env and add your GOOGLE_API_KEY
```

---

## 🔑 API Key Setup

Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

```bash
export GOOGLE_API_KEY=your_key_here
```

Or add to `.env`:
```
GOOGLE_API_KEY=your_key_here
```

---

## 🏃 Quick Start

### Desktop GUI (Recommended)
```bash
python gui.py
```

The GUI provides a full visual workflow:
- **Sidebar** — upload case documents & evidence, select jurisdiction, describe your situation
- **Progress tab** — live log of the AI pipeline
- **Generated Documents tab** — open each document directly from the app
- **Find & Edit tab** — find/replace in Word docs, AI post-processing fix, and one-click Preset Legal Updates (e.g. remove ambiguous diagnostic claims, restore statute citations)

### Interactive Demo (CLI)
```bash
python main.py
```

### Quick Research Query
```bash
python main.py --research "What are the elements of breach of contract?"
```

Research results are automatically exported as a formatted HTML report and opened in your browser. You can also export as Markdown or JSON:
```bash
python main.py --research "statute of limitations Nebraska guardianship" --state NE --export html
python main.py --research "..." --export md
python main.py --research "..." --export json
python main.py --research "..." --export all   # HTML + MD + JSON
```

### With State Jurisdiction
```bash
python main.py --research "personal injury statute of limitations" --state CA
```

### Evidence Analysis & Document Generation (CLI)
```bash
python main.py --evidence --state NE \
    --situation "Description of your case…" \
    --files evidence1.pdf recording.mp3
```

### System Info
```bash
python main.py --info
```

---

## 🗺️ Adding State Laws (Customizable Infrastructure)

### Interactive Wizard
```bash
python scripts/add_state.py --interactive
```

### Generate Template
```bash
python scripts/add_state.py --template FL Florida
```

### Import from JSON
```bash
python scripts/add_state.py --import data/state_configs/fl_config.json
```

### List Configured States
```bash
python scripts/add_state.py --list
```

---

## 📁 Project Structure

```
rapcorp-legal-ai-system/
├── gui.py                       # Desktop GUI entry point
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
├── .env.template               # Environment template
│
├── configs/
│   └── config.py               # Master configuration
│
├── src/
│   ├── legal_ai_system.py      # Main orchestrator
│   ├── core/
│   │   └── gemini_client.py    # Gemini API client
│   ├── state_laws/
│   │   └── state_manager.py    # State law infrastructure
│   ├── swarm/
│   │   └── knowledge_swarm.py  # 15-agent swarm
│   ├── reasoning/
│   │   └── reasoning_engine.py # IRAC/CREAC engine
│   └── ...
│
├── scripts/
│   └── add_state.py            # State addition tool
│
├── data/
│   ├── state_configs/          # State JSON configs
│   └── embeddings/             # Vector embeddings
│
└── templates/                  # Document templates
```

---

## 🎯 Gemini Model Selection

The system automatically selects the optimal model based on task:

| Task Type | Model | Cost |
|-----------|-------|------|
| Classification, Simple Lookup | `gemini-flash-lite-latest` | ~$0.01/query |
| Retrieval, Summarization, Drafting | `gemini-flash-latest` | ~$0.02/query |
| IRAC Analysis, Complex Reasoning | `gemini-pro-latest` | ~$0.05/query |

### Manual Override
```python
result = await client.generate(
    prompt="...",
    model_override="gemini-pro-latest"  # Force Pro model
)
```

---

## 🗺️ State Law Configuration

### Pre-Configured States
- **California (CA)** - Full configuration
- **New York (NY)** - Full configuration
- **Texas (TX)** - Full configuration
- **Federal (FED)** - Federal law

### State Config JSON Structure
```json
{
  "state": "FL",
  "state_name": "Florida",
  "statute_of_limitations": {
    "personal_injury": 4,
    "contract_written": 5,
    "medical_malpractice": 2
  },
  "court_hierarchy": [
    "Supreme Court of Florida",
    "Florida District Courts of Appeal",
    "Florida Circuit Courts"
  ],
  "special_rules": {
    "anti_slapp": true,
    "no_fault_auto": true,
    "homestead_protection": true
  },
  "is_active": true
}
```

### Programmatic State Addition
```python
from src.state_laws.state_manager import StateLawManager
from configs.config import create_config, USState

config = create_config()
manager = StateLawManager(config)

# Add Florida with custom SOL
florida = manager.add_state(
    USState.FLORIDA,
    "Florida",
    template="common_law",
    statute_of_limitations={
        "personal_injury": 4,  # Florida has 4 years!
        "contract_written": 5,
    },
    special_rules={
        "no_fault_auto": True,
        "homestead_protection": True
    }
)

# Save to JSON
manager.save_state(USState.FLORIDA)
```

---

## 🧠 Usage Examples

### Basic Research
```python
import asyncio
from src.legal_ai_system import create_legal_ai_system
from configs.config import USState, LegalDomain

async def main():
    system = await create_legal_ai_system()
    
    result = await system.research(
        query="What are the elements of breach of contract?",
        state=USState.CALIFORNIA,
        domain=LegalDomain.CONTRACT,
        mode="standard"
    )
    
    print(result["response"])

asyncio.run(main())
```

### IRAC Analysis
```python
analysis = await system.analyze_irac(
    issue="Whether oral promise to pay $5000 is enforceable",
    facts={
        "promise_amount": 5000,
        "written": False,
        "consideration": "services rendered"
    },
    state=USState.CALIFORNIA
)

print(f"Issue: {analysis['issue']}")
print(f"Rule: {analysis['rule']}")
print(f"Application: {analysis['application']}")
print(f"Conclusion: {analysis['conclusion']}")
```

### Statute of Limitations Check
```python
from datetime import datetime

sol = system.get_sol(
    state=USState.FLORIDA,
    claim_type="personal_injury",
    accrual_date=datetime(2023, 6, 15)
)

print(f"Deadline: {sol['deadline']}")
print(f"Days Remaining: {sol['days_remaining']}")
print(f"Expired: {sol['is_expired']}")
```

---

## 💰 Cost Management

```python
# Check budget status
summary = system.get_cost_summary()
print(f"Today's Cost: ${summary['today_cost']:.4f}")
print(f"Budget Remaining: ${summary['budget_status']['daily']['remaining']:.2f}")
```

Configure in `.env`:
```
DAILY_BUDGET_USD=5.00
MONTHLY_BUDGET_USD=100.00
```

---

## 📋 Available Legal Domains

- `contract` - Contract law
- `tort` - Tort/personal injury
- `criminal` - Criminal law
- `family` - Family law
- `employment` - Employment law
- `real_estate` - Real estate
- `ip` - Intellectual property
- `corporate` - Corporate law
- `bankruptcy` - Bankruptcy
- `tax` - Tax law
- And more...

---

## ⚖️ Disclaimer

This system is for **educational and research purposes only**. It does not constitute legal advice. Always consult with a licensed attorney in your jurisdiction for legal matters.

---

## 📄 License

Proprietary - RAPCorp (Renaissance of American Physics and Astronomy)

---

## 🤝 Contributing

Contact RAPCorp for contribution guidelines.
