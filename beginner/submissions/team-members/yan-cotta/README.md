# FinResearch AI - Educational Demo Collection

**Beginner Track: Learning Agentic AI for Financial Research**

This directory contains a progressive series of educational demonstrations designed to teach the fundamentals of agentic AI systems. Each module builds upon the previous, introducing new concepts and frameworks used in modern AI agent development.

---

## Prerequisites

Before running any demos, ensure you have:

1. Python 3.10 or higher installed
2. A valid OpenAI API key
3. Project dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```
4. Environment configured:
   ```bash
   # Create .env file in project root with:
   OPENAI_API_KEY=your-api-key-here
   ```

---

## Directory Structure

```
yan-cotta/
├── README.md                    # This file
├── docs/                        # Architecture diagrams and reference materials
│
├── 01_hello_world/              # Module 1: First Agent
│   └── demo_simple_agent.py
│
├── 02_tools_and_data/           # Module 2: Tools and External Data
│   ├── demo_openai_sdk.py
│   ├── demo_financial_crew.py
│   └── week2_data_fetcher.py
│
├── 03_agent_patterns/           # Module 3: Agent Architectures
│   ├── demo_langchain_agent.py
│   └── week2_agents.py
│
└── 04_advanced_concepts/        # Module 4: Production Patterns
    ├── demo_openai_assistants.py
    └── demo_langgraph_workflow.py
```

---

## Learning Path

### Module 1: Hello World - Your First Agent

**Directory:** `01_hello_world/`

| File | Framework | Concepts Covered |
|------|-----------|------------------|
| `demo_simple_agent.py` | CrewAI | Agent, Task, Crew fundamentals |

**Learning Objectives:**
- Understand the three atomic units of CrewAI: Agent, Task, Crew
- Learn how to define an agent's role, goal, and backstory
- See how tasks are assigned and executed
- Observe verbose output to understand agent reasoning

**Run:**
```bash
python 01_hello_world/demo_simple_agent.py
```

---

### Module 2: Tools and External Data

**Directory:** `02_tools_and_data/`

| File | Framework | Concepts Covered |
|------|-----------|------------------|
| `demo_openai_sdk.py` | OpenAI SDK | Function calling, tool definitions |
| `demo_financial_crew.py` | CrewAI | Custom tools, Yahoo Finance integration |
| `week2_data_fetcher.py` | Python | Data fetching patterns, yfinance usage |

**Learning Objectives:**
- Understand how LLMs interact with external tools
- Learn OpenAI's function calling mechanism
- Create custom tools that fetch real market data
- Handle API responses and error cases

**Run:**
```bash
python 02_tools_and_data/demo_openai_sdk.py
python 02_tools_and_data/demo_financial_crew.py
```

---

### Module 3: Agent Patterns and Architectures

**Directory:** `03_agent_patterns/`

| File | Framework | Concepts Covered |
|------|-----------|------------------|
| `demo_langchain_agent.py` | LangChain + LangGraph | ReAct pattern, tool-calling agents |
| `week2_agents.py` | CrewAI | Multi-agent systems, role separation |

**Learning Objectives:**
- Understand the ReAct (Reasoning + Acting) pattern
- Learn why specialized agents outperform monolithic ones
- See how agents collaborate through shared context
- Compare different agent orchestration approaches

**Key Concept - Why Multiple Agents?**

Single "super agents" with all tools suffer from:
1. Context overload - too many capabilities cause confusion
2. Hallucination risk - agents may confuse tool capabilities
3. Debugging difficulty - hard to isolate issues

Specialized agents provide:
- Clear responsibilities and focused prompts
- Reduced hallucination through role constraints
- Easier testing and maintenance

**Run:**
```bash
python 03_agent_patterns/demo_langchain_agent.py
python 03_agent_patterns/week2_agents.py AAPL
```

---

### Module 4: Advanced Concepts

**Directory:** `04_advanced_concepts/`

| File | Framework | Concepts Covered |
|------|-----------|------------------|
| `demo_openai_assistants.py` | OpenAI Assistants API | Persistent assistants, managed threads |
| `demo_langgraph_workflow.py` | LangGraph | StateGraph, conditional routing, workflows |

**Learning Objectives:**
- Understand stateful conversation management
- Learn graph-based agent orchestration
- Implement conditional routing between agents
- Design multi-step research workflows

**LangGraph Architecture:**

```
START
  │
  ▼
┌─────────────────┐
│  Data Collector │ ── Fetches stock data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Analyst Agent  │ ── Analyzes collected data
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  CONDITIONAL ROUTING        │
│  ├── More research needed → Researcher
│  └── Ready for report → Report Writer
└─────────────────────────────┘
```

**Run:**
```bash
python 04_advanced_concepts/demo_openai_assistants.py
python 04_advanced_concepts/demo_langgraph_workflow.py TSLA
```

---

## Documentation

The `docs/` directory contains:

| File | Description |
|------|-------------|
| `financial_multiagentic_system_architecture.pdf` | System architecture overview |
| `*.png` | Architecture diagrams and flowcharts |

---

## Framework Comparison

| Framework | Best For | Complexity | Key Feature |
|-----------|----------|------------|-------------|
| OpenAI SDK | Simple function calling | Low | Native tool support |
| CrewAI | Multi-agent teams | Medium | Role-based agents |
| LangChain | Flexible chains | Medium | Extensive integrations |
| LangGraph | Complex workflows | High | Graph-based orchestration |
| Assistants API | Stateful conversations | Medium | Managed state |

---

## Troubleshooting

### "OPENAI_API_KEY not found"
Ensure your `.env` file exists in the project root and contains a valid key.

### "Module not found" errors
Install dependencies: `pip install -r requirements.txt`

### Rate limit errors
OpenAI has rate limits. Wait a moment and retry, or use a different model.

### yfinance returns empty data
- Verify the ticker symbol is valid
- Check your internet connection
- Some tickers may not have all data fields

---

## Next Steps

After completing these tutorials, proceed to the **Advanced Track** located at:
```
advanced/submissions/team-members/yan-cotta/
```

The advanced track contains a production-grade, hierarchical multi-agent system with:
- Proper Python package structure
- Configuration management with YAML and Pydantic
- Docker containerization
- Comprehensive test coverage
- CLI interface

---

## Author

**Yan Cotta**  
FinResearch AI Project  
December 2024
