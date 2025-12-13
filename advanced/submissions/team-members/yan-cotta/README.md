# FinResearch AI - Advanced Track

## Multi-Agent Financial Research System

A production-grade, hierarchical multi-agent system for automated financial research, built with CrewAI. This implementation serves as the gold standard reference for the FinResearch AI project.

---

## System Architecture

```
                              INPUT
                    ┌─────────────────────┐
                    │   CLI (main.py)     │
                    │   Config (.env)     │
                    └──────────┬──────────┘
                               │
                               ▼
                         ORCHESTRATION
                    ┌─────────────────────┐
                    │   FinResearchCrew   │
                    │     (crew.py)       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Manager Agent     │
                    │   (Orchestrator)    │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │ Researcher  │    │  Analyst    │    │  Reporter   │
     │ Qualitative │    │ Quantitative│    │  Synthesis  │
     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
            │                  │                  │
            ▼                  ▼                  ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │ NewsSearch  │    │ Financial   │    │   Memory    │
     │   Tool      │    │  DataTool   │    │    Tool     │
     │ (DuckDuckGo)│    │  (yfinance) │    │  (ChromaDB) │
     └─────────────┘    └─────────────┘    └─────────────┘
                               │
                               ▼
                           OUTPUT
                    ┌─────────────────────┐
                    │  Markdown Report    │
                    │    (./reports/)     │
                    └─────────────────────┘
```

---

## Directory Structure

```
yan-cotta/
├── main.py                 # CLI entry point
├── pyproject.toml          # Project metadata and dependencies
├── Dockerfile              # Container configuration
├── .dockerignore           # Docker build exclusions
├── README.md               # This documentation
│
├── src/
│   ├── __init__.py
│   ├── crew.py             # Crew orchestration logic
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py     # Pydantic settings management
│   │   ├── agents.yaml     # Agent personas and prompts
│   │   └── tasks.yaml      # Task templates
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py         # Base factory and utilities
│   │   ├── manager.py      # Manager agent (orchestrator)
│   │   ├── researcher.py   # Researcher agent (qualitative)
│   │   ├── analyst.py      # Analyst agent (quantitative)
│   │   └── reporter.py     # Reporter agent (synthesis)
│   │
│   └── tools/
│       ├── __init__.py
│       ├── base.py         # Base tool classes
│       ├── financial_data.py   # Yahoo Finance wrapper
│       ├── news_search.py      # DuckDuckGo wrapper
│       └── memory.py           # ChromaDB memory tool
│
└── tests/
    ├── __init__.py
    └── test_financial_tool.py  # Tool unit tests
```

---

## Module Overview

### Configuration (`src/config/`)

| File | Purpose |
|------|---------|
| `settings.py` | Pydantic-based settings with environment variable support |
| `agents.yaml` | Agent roles, goals, and backstories (prompts) |
| `tasks.yaml` | Task description templates with placeholders |

### Tools (`src/tools/`)

| Tool | Data Source | Purpose |
|------|-------------|---------|
| `FinancialDataTool` | Yahoo Finance | Stock prices, valuation metrics, fundamentals |
| `NewsSearchTool` | DuckDuckGo | News articles with source verification |
| `MemoryTool` | ChromaDB | Persistent vector memory for agent collaboration |

### Agents (`src/agents/`)

| Agent | Role | Temperature | Tools |
|-------|------|-------------|-------|
| Manager | Orchestration and delegation | 0.1 | Memory |
| Researcher | Qualitative analysis | 0.7 | News, Memory |
| Analyst | Quantitative analysis | 0.0 | Financial, Memory |
| Reporter | Report synthesis | 0.5 | Memory |

### Crew Orchestration (`src/crew.py`)

Provides two execution modes:

- `FinResearchCrew`: Hierarchical process with Manager delegation
- `SequentialFinResearchCrew`: Linear task execution (for debugging)

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Virtual environment (recommended)

### Installation

```bash
# Navigate to this directory
cd advanced/submissions/team-members/yan-cotta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key

# Optional overrides
FINRESEARCH_MANAGER_MODEL=gpt-4o-mini
FINRESEARCH_WORKER_MODEL=gpt-3.5-turbo
FINRESEARCH_LOG_LEVEL=INFO
FINRESEARCH_OUTPUT_DIR=./reports
```

---

## Usage

### Command Line Interface

```bash
# Research Apple Inc
python main.py AAPL

# Research with company name
python main.py TSLA --name "Tesla Inc"

# Use sequential process (simpler, for debugging)
python main.py MSFT --sequential

# Custom output filename
python main.py GOOGL --output google_research.md

# Verbose mode (see agent reasoning)
python main.py NVDA --verbose
```

### CLI Options

```
positional arguments:
  ticker                Stock ticker symbol (e.g., AAPL, TSLA)

optional arguments:
  -n, --name            Company name (defaults to ticker)
  -o, --output          Output filename for report
  -s, --sequential      Use sequential instead of hierarchical process
  -v, --verbose         Enable verbose agent output
  -q, --quiet           Suppress banner and progress
  --log-level           Logging level (DEBUG/INFO/WARNING/ERROR)
  --log-file            Log to file instead of stdout
  --dry-run             Validate config without running
```

### Python API

```python
from src.crew import FinResearchCrew

# Create and run crew
crew = FinResearchCrew(
    ticker="AAPL",
    company_name="Apple Inc",
    verbose=True
)

# Execute research
report = crew.run()

# Save report
path = crew.save_report(report)
print(f"Report saved to: {path}")
```

### Docker

```bash
# Build image
docker build -t finresearch-advanced .

# Run research
docker run -e OPENAI_API_KEY=sk-... finresearch-advanced AAPL
```

---

## Workflow

```
User                Manager             Researcher          Analyst             Reporter            Memory
 │                     │                    │                   │                   │                  │
 │  Research AAPL      │                    │                   │                   │                  │
 │────────────────────>│                    │                   │                   │                  │
 │                     │                    │                   │                   │                  │
 │                     │  Find news         │                   │                   │                  │
 │                     │───────────────────>│                   │                   │                  │
 │                     │                    │                   │                   │                  │
 │                     │                    │  Save findings    │                   │                  │
 │                     │                    │─────────────────────────────────────────────────────────>│
 │                     │                    │                   │                   │                  │
 │                     │  Research done     │                   │                   │                  │
 │                     │<───────────────────│                   │                   │                  │
 │                     │                    │                   │                   │                  │
 │                     │  Get financials    │                   │                   │                  │
 │                     │──────────────────────────────────────>│                   │                  │
 │                     │                    │                   │                   │                  │
 │                     │                    │                   │  Save metrics     │                  │
 │                     │                    │                   │──────────────────────────────────────>│
 │                     │                    │                   │                   │                  │
 │                     │  Analysis done     │                   │                   │                  │
 │                     │<──────────────────────────────────────│                   │                  │
 │                     │                    │                   │                   │                  │
 │                     │  Create report     │                   │                   │                  │
 │                     │─────────────────────────────────────────────────────────>│                  │
 │                     │                    │                   │                   │                  │
 │                     │                    │                   │                   │  Get all data    │
 │                     │                    │                   │                   │─────────────────>│
 │                     │                    │                   │                   │                  │
 │                     │  Report ready      │                   │                   │                  │
 │                     │<─────────────────────────────────────────────────────────│                  │
 │                     │                    │                   │                   │                  │
 │  Final report       │                    │                   │                   │                  │
 │<────────────────────│                    │                   │                   │                  │
 │                     │                    │                   │                   │                  │
```

---

## Output Example

Reports are saved to `./reports/` with the format:

```
report_AAPL_20241212_143052.md
```

Sample report structure:

```markdown
# Investment Research Report: Apple Inc (AAPL)

Generated: 2024-12-12 14:30:52 UTC

## Executive Summary
...

## Recent Developments
...

## Financial Analysis
...

## Investment Considerations
...

## Risk Factors
...

## Disclaimer
This report is for informational purposes only...
```

---

## Design Decisions

### Why Hierarchical Process?

The Manager agent coordinates work, ensuring:

- Proper task sequencing
- Quality control before final output
- Efficient delegation to specialists

### Why Separate Tools per Agent?

- **Prevents hallucination**: Analyst cannot fabricate news articles
- **Clear responsibilities**: Each agent has focused expertise
- **Easier debugging**: Issues are isolated to specific agents

### Why ChromaDB Memory?

- Enables agent collaboration without direct communication
- Persists context across task boundaries
- Supports semantic retrieval for relevant information

---

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_financial_tool.py -v
```

---

## Troubleshooting

### OPENAI_API_KEY not configured

Ensure your `.env` file is in the project root and contains a valid key.

### No data found for ticker

Verify the ticker symbol is valid on Yahoo Finance.

### ChromaDB not installed

Run: `pip install chromadb`

### Agent seems stuck

Use `--sequential` mode for simpler execution and easier debugging.

---

## License

MIT License - See project root for details.

---

**Author:** Yan Cotta

**Version:** 1.0.0

**Last Updated:** December 2025
