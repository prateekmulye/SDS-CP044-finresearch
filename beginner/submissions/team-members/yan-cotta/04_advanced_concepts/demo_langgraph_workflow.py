#!/usr/bin/env python3
"""
================================================================================
LANGGRAPH MULTI-AGENT WORKFLOW DEMO - ADVANCED FINANCIAL RESEARCH
================================================================================

Welcome to the Advanced LangGraph Demo! This demonstrates how to build
sophisticated multi-agent workflows using LangGraph's StateGraph pattern.

🎯 LEARNING OBJECTIVES:
    1. Understand LangGraph's graph-based agent orchestration
    2. Learn the StateGraph pattern for stateful workflows
    3. See how to implement conditional routing between agents
    4. Master the concept of agent handoffs and message passing

📚 WHAT IS LANGGRAPH?
    LangGraph is a library for building stateful, multi-agent applications
    with LLMs. It extends LangChain with graph-based orchestration:
    
    - NODES: Individual agents or functions that process state
    - EDGES: Connections that define flow between nodes
    - STATE: Shared data that flows through the graph
    - CONDITIONAL EDGES: Dynamic routing based on state

🏗️ ARCHITECTURE OF THIS DEMO:
    
    ┌─────────────────┐
    │   START         │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Data Collector │ ──── Fetches stock data using yfinance
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Analyst Agent  │ ──── Analyzes the collected data
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────┐
    │  CONDITIONAL ROUTING                      │
    │  ├── If more research needed → Researcher│
    │  └── If ready for report → Report Writer │
    └────────┬──────────────────────┬──────────┘
             │                      │
             ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐
    │   Researcher    │    │  Report Writer  │
    └────────┬────────┘    └────────┬────────┘
             │                      │
             └──────────┬───────────┘
                        ▼
               ┌─────────────────┐
               │      END        │
               └─────────────────┘

💡 REAL-WORLD APPLICATIONS:
    - Multi-stage research pipelines
    - Investment analysis workflows
    - Risk assessment systems
    - Automated financial reporting

Author: FinResearch AI Team (Yan Cotta)
Level: ADVANCED - Assumes familiarity with LangChain and async Python
Framework: LangGraph
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND SETUP
# ==============================================================================

import os
import sys
from datetime import datetime
from typing import Annotated, TypedDict, Literal

# Environment variable management - loads API keys from .env file
# This keeps sensitive credentials out of source code!
from dotenv import load_dotenv

# LangGraph imports for graph-based workflows
# - StateGraph: The main class for building stateful graphs
# - END: A special constant marking the end of the workflow
from langgraph.graph import StateGraph, END

# LangChain imports for LLM interaction
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Yahoo Finance for real market data
import yfinance as yf

# ==============================================================================
# EDUCATIONAL PRINT HELPERS
# ==============================================================================

def print_header(title: str) -> None:
    """Print a formatted section header for educational clarity."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_step(step: str, description: str) -> None:
    """Print a workflow step with visual indicator."""
    print(f"\n🔄 STEP: {step}")
    print(f"   {description}")
    print("-" * 50)

def print_node_entry(node_name: str) -> None:
    """Print when entering a graph node."""
    print(f"\n📍 ENTERING NODE: [{node_name}]")

def print_node_exit(node_name: str) -> None:
    """Print when exiting a graph node."""
    print(f"✅ EXITING NODE: [{node_name}]")

# ==============================================================================
# SECTION 2: ENVIRONMENT CONFIGURATION
# ==============================================================================

def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    EDUCATIONAL NOTE:
    -----------------
    LangGraph agents need API keys to function. This function:
    1. Loads the .env file from the project root
    2. Checks if OPENAI_API_KEY is set
    3. Provides helpful error messages if missing
    
    SECURITY BEST PRACTICE:
    - Never hardcode API keys
    - Use .env files for local development
    - Use secrets management in production
    """
    
    print_header("ENVIRONMENT VALIDATION")
    
    # Load environment variables from .env file
    # The .env file should be in the project root, so we traverse up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find .env file by traversing up the directory tree
    for _ in range(5):  # Check up to 5 levels up
        env_path = os.path.join(current_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✅ Loaded .env from: {env_path}")
            break
        current_dir = os.path.dirname(current_dir)
    else:
        print("⚠️  No .env file found, using system environment variables")
    
    # Check for required API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found!")
        print("\n📝 TO FIX: Create a .env file in the project root with:")
        print("   OPENAI_API_KEY=your-openai-api-key-here")
        return False
    
    # Validate key format (basic check)
    if not api_key.startswith("sk-"):
        print("\n⚠️  WARNING: API key format looks unusual (should start with 'sk-')")
    
    print(f"✅ OPENAI_API_KEY is set (starts with: {api_key[:10]}...)")
    return True

# ==============================================================================
# SECTION 3: STATE DEFINITION
# ==============================================================================
"""
EDUCATIONAL NOTE: STATE IN LANGGRAPH
====================================

The State is the heart of LangGraph workflows. It's a TypedDict that:
1. Defines ALL data that flows through the graph
2. Is passed to every node function
3. Gets updated by each node's return value
4. Persists across the entire workflow

DESIGN PRINCIPLES:
- Keep state flat and simple
- Include all data nodes might need
- Track workflow progress with status fields
- Store intermediate results for debugging
"""

class FinancialResearchState(TypedDict):
    """
    The shared state that flows through our financial research workflow.
    
    FIELDS EXPLAINED:
    ----------------
    - ticker: The stock symbol being researched (e.g., "TSLA")
    - stock_data: Raw data fetched from yfinance
    - analysis: The analyst's interpretation of the data
    - needs_more_research: Flag for conditional routing
    - research_notes: Additional findings from the researcher
    - final_report: The completed research report
    - current_step: Tracks progress through the workflow
    - messages: History of all LLM interactions (for debugging)
    """
    ticker: str
    stock_data: dict
    analysis: str
    needs_more_research: bool
    research_notes: str
    final_report: str
    current_step: str
    messages: list

# ==============================================================================
# SECTION 4: DATA COLLECTION NODE
# ==============================================================================

def data_collector_node(state: FinancialResearchState) -> FinancialResearchState:
    """
    Node 1: Collect stock data from Yahoo Finance.
    
    EDUCATIONAL NOTE:
    -----------------
    This node demonstrates PURE DATA COLLECTION without LLM involvement.
    Not every node needs an LLM - sometimes you just need to fetch data!
    
    PATTERN: Data Collector Node
    - Input: Stock ticker from state
    - Process: Fetch real data from external API
    - Output: Update state with fetched data
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state with stock_data populated
    """
    
    print_node_entry("DATA_COLLECTOR")
    
    ticker = state["ticker"]
    print(f"   📊 Fetching data for: {ticker}")
    
    try:
        # Create yfinance Ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch comprehensive stock information
        info = stock.info
        
        # Extract key metrics for analysis
        stock_data = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "previous_close": info.get("previousClose", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "volume": info.get("volume", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
            "beta": info.get("beta", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A")[:500] + "..." if info.get("longBusinessSummary") else "N/A"
        }
        
        print(f"   ✅ Successfully fetched data for {stock_data['name']}")
        print(f"   💰 Current Price: ${stock_data['current_price']}")
        
    except Exception as e:
        print(f"   ❌ Error fetching data: {e}")
        stock_data = {"error": str(e), "ticker": ticker}
    
    print_node_exit("DATA_COLLECTOR")
    
    # Return updated state
    # IMPORTANT: We return a dict with ONLY the fields we want to update
    # LangGraph will merge this with the existing state
    return {
        **state,
        "stock_data": stock_data,
        "current_step": "data_collected"
    }

# ==============================================================================
# SECTION 5: ANALYST NODE
# ==============================================================================

def analyst_node(state: FinancialResearchState) -> FinancialResearchState:
    """
    Node 2: Analyze the collected stock data using an LLM.
    
    EDUCATIONAL NOTE:
    -----------------
    This node demonstrates LLM-POWERED ANALYSIS.
    We take raw data and use an LLM to interpret it.
    
    PATTERN: Analyst Node
    - Input: Raw data from previous node
    - Process: Use LLM to analyze and interpret
    - Output: Analysis text + decision on next steps
    
    KEY CONCEPT: This node also sets `needs_more_research` flag,
    which controls CONDITIONAL ROUTING to the next node.
    """
    
    print_node_entry("ANALYST")
    
    stock_data = state["stock_data"]
    
    # Check for data errors
    if "error" in stock_data:
        print(f"   ⚠️  Cannot analyze - data collection failed")
        return {
            **state,
            "analysis": f"Analysis failed: {stock_data['error']}",
            "needs_more_research": False,
            "current_step": "analysis_failed"
        }
    
    # Initialize the LLM
    # EDUCATIONAL NOTE: We use a specific temperature for analysis
    # Lower temperature = more consistent, focused responses
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Cost-effective model for analysis
        temperature=0.3       # Lower temp for more factual analysis
    )
    
    # Construct the analysis prompt
    # BEST PRACTICE: Use clear, structured prompts with specific instructions
    analysis_prompt = f"""
    You are a financial analyst. Analyze the following stock data and provide:
    1. A brief summary of the company's current position
    2. Key metrics that stand out (positive or negative)
    3. A preliminary assessment (bullish, bearish, or neutral)
    
    Also indicate if MORE RESEARCH is needed. More research is needed if:
    - The PE ratio is significantly different from industry average
    - The stock is near its 52-week high or low
    - There are unusual volume patterns
    
    Stock Data:
    -----------
    Company: {stock_data.get('name')}
    Ticker: {stock_data.get('ticker')}
    Current Price: ${stock_data.get('current_price')}
    Previous Close: ${stock_data.get('previous_close')}
    Market Cap: {stock_data.get('market_cap')}
    P/E Ratio: {stock_data.get('pe_ratio')}
    52-Week High: ${stock_data.get('52_week_high')}
    52-Week Low: ${stock_data.get('52_week_low')}
    Volume: {stock_data.get('volume')}
    Avg Volume: {stock_data.get('avg_volume')}
    Beta: {stock_data.get('beta')}
    Sector: {stock_data.get('sector')}
    Industry: {stock_data.get('industry')}
    
    End your response with exactly one of these lines:
    NEEDS_MORE_RESEARCH: YES
    or
    NEEDS_MORE_RESEARCH: NO
    """
    
    print("   🤖 Sending data to Analyst LLM...")
    
    # Call the LLM
    response = llm.invoke([
        SystemMessage(content="You are an expert financial analyst."),
        HumanMessage(content=analysis_prompt)
    ])
    
    analysis_text = response.content
    print("   📝 Analysis received")
    
    # Parse the decision for conditional routing
    # EDUCATIONAL NOTE: We extract structured data from LLM output
    needs_more = "NEEDS_MORE_RESEARCH: YES" in analysis_text.upper()
    
    print(f"   🔍 More research needed: {needs_more}")
    
    print_node_exit("ANALYST")
    
    return {
        **state,
        "analysis": analysis_text,
        "needs_more_research": needs_more,
        "current_step": "analysis_complete",
        "messages": state.get("messages", []) + [
            {"role": "analyst", "content": analysis_text}
        ]
    }

# ==============================================================================
# SECTION 6: RESEARCHER NODE
# ==============================================================================

def researcher_node(state: FinancialResearchState) -> FinancialResearchState:
    """
    Node 3A: Conduct additional research when needed.
    
    EDUCATIONAL NOTE:
    -----------------
    This node is reached via CONDITIONAL ROUTING.
    It only executes if the analyst determined more research is needed.
    
    PATTERN: Deep Research Node
    - Input: Previous analysis + original data
    - Process: Dig deeper into specific concerns
    - Output: Additional research notes
    
    In a production system, this might:
    - Query additional data sources
    - Look up news articles
    - Compare with competitors
    - Analyze historical trends
    """
    
    print_node_entry("RESEARCHER")
    
    stock_data = state["stock_data"]
    analysis = state["analysis"]
    
    # Initialize LLM with slightly higher temperature for creative research
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5  # Slightly more creative for research
    )
    
    research_prompt = f"""
    Based on the initial analysis, conduct deeper research on this stock.
    
    INITIAL ANALYSIS:
    {analysis}
    
    STOCK DATA:
    - Company: {stock_data.get('name')}
    - Sector: {stock_data.get('sector')}
    - Industry: {stock_data.get('industry')}
    - P/E Ratio: {stock_data.get('pe_ratio')}
    - Beta: {stock_data.get('beta')}
    
    Please provide:
    1. Industry context and comparison
    2. Risk factors to consider
    3. Potential catalysts (positive or negative)
    4. Key questions an investor should ask
    
    Keep your response focused and actionable.
    """
    
    print("   🔬 Conducting deep research...")
    
    response = llm.invoke([
        SystemMessage(content="You are a senior equity researcher with 20 years of experience."),
        HumanMessage(content=research_prompt)
    ])
    
    research_notes = response.content
    print("   📋 Research complete")
    
    print_node_exit("RESEARCHER")
    
    return {
        **state,
        "research_notes": research_notes,
        "current_step": "research_complete",
        "messages": state.get("messages", []) + [
            {"role": "researcher", "content": research_notes}
        ]
    }

# ==============================================================================
# SECTION 7: REPORT WRITER NODE
# ==============================================================================

def report_writer_node(state: FinancialResearchState) -> FinancialResearchState:
    """
    Node 4: Generate the final research report.
    
    EDUCATIONAL NOTE:
    -----------------
    This is the TERMINAL NODE - it produces the final output.
    It synthesizes all previous work into a cohesive report.
    
    PATTERN: Report Writer Node
    - Input: All accumulated state (data, analysis, research)
    - Process: Synthesize into professional report
    - Output: Final formatted report
    
    KEY CONCEPT: This node must handle cases where:
    - Research was done (has research_notes)
    - Research was skipped (no research_notes)
    """
    
    print_node_entry("REPORT_WRITER")
    
    stock_data = state["stock_data"]
    analysis = state["analysis"]
    research_notes = state.get("research_notes", "")
    
    # Initialize LLM with low temperature for consistent formatting
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2  # Low temp for consistent reports
    )
    
    # Build the report prompt based on available information
    research_section = f"""
    ADDITIONAL RESEARCH FINDINGS:
    {research_notes}
    """ if research_notes else "No additional research was conducted."
    
    report_prompt = f"""
    Generate a professional financial research report for {stock_data.get('name')} ({stock_data.get('ticker')}).
    
    STOCK DATA:
    - Current Price: ${stock_data.get('current_price')}
    - Market Cap: {stock_data.get('market_cap')}
    - P/E Ratio: {stock_data.get('pe_ratio')}
    - 52-Week Range: ${stock_data.get('52_week_low')} - ${stock_data.get('52_week_high')}
    - Sector: {stock_data.get('sector')}
    
    ANALYST ASSESSMENT:
    {analysis}
    
    {research_section}
    
    Format the report with these sections:
    1. EXECUTIVE SUMMARY (2-3 sentences)
    2. KEY METRICS ANALYSIS
    3. RISK ASSESSMENT
    4. RECOMMENDATION (Buy/Hold/Sell with rationale)
    5. DISCLAIMER
    
    Keep it professional and concise.
    Report Date: {datetime.now().strftime('%Y-%m-%d')}
    """
    
    print("   📄 Generating final report...")
    
    response = llm.invoke([
        SystemMessage(content="You are a professional financial report writer at a major investment bank."),
        HumanMessage(content=report_prompt)
    ])
    
    final_report = response.content
    print("   ✅ Report generated")
    
    print_node_exit("REPORT_WRITER")
    
    return {
        **state,
        "final_report": final_report,
        "current_step": "report_complete",
        "messages": state.get("messages", []) + [
            {"role": "report_writer", "content": final_report}
        ]
    }

# ==============================================================================
# SECTION 8: CONDITIONAL ROUTING FUNCTION
# ==============================================================================

def should_research(state: FinancialResearchState) -> Literal["researcher", "report_writer"]:
    """
    Conditional routing function that determines the next node.
    
    EDUCATIONAL NOTE:
    -----------------
    This is a ROUTING FUNCTION - it doesn't modify state, it just decides
    which node to execute next based on the current state.
    
    PATTERN: Conditional Edges
    - LangGraph allows dynamic routing based on state
    - The function returns the NAME of the next node
    - This enables complex decision trees in workflows
    
    Returns:
        "researcher" if more research is needed
        "report_writer" if ready to generate report
    """
    
    print("\n🔀 CONDITIONAL ROUTING DECISION:")
    
    needs_more = state.get("needs_more_research", False)
    
    if needs_more:
        print("   → Routing to RESEARCHER (more research needed)")
        return "researcher"
    else:
        print("   → Routing to REPORT_WRITER (ready for report)")
        return "report_writer"

# ==============================================================================
# SECTION 9: GRAPH CONSTRUCTION
# ==============================================================================

def build_research_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for financial research.
    
    EDUCATIONAL NOTE:
    -----------------
    This function demonstrates GRAPH CONSTRUCTION in LangGraph:
    
    1. Create a StateGraph with your State type
    2. Add nodes (functions that process state)
    3. Add edges (connections between nodes)
    4. Add conditional edges (dynamic routing)
    5. Set entry point
    6. Compile the graph
    
    ANATOMY OF A LANGGRAPH:
    
        StateGraph(State)           # Initialize with state type
            │
            ├── add_node("name", fn) # Add processing nodes
            │
            ├── add_edge("a", "b")   # Static connections
            │
            ├── add_conditional_edges(
            │       "node",           # From this node
            │       routing_fn,       # Call this function
            │       {"choice": "target"} # Map choices to nodes
            │   )
            │
            ├── set_entry_point("start") # Where to begin
            │
            └── compile()             # Create runnable graph
    
    Returns:
        Compiled StateGraph ready to execute
    """
    
    print_header("BUILDING LANGGRAPH WORKFLOW")
    
    # Step 1: Initialize the StateGraph with our state type
    print("   1️⃣  Initializing StateGraph with FinancialResearchState")
    workflow = StateGraph(FinancialResearchState)
    
    # Step 2: Add all nodes to the graph
    print("   2️⃣  Adding nodes:")
    
    workflow.add_node("data_collector", data_collector_node)
    print("       ✓ data_collector")
    
    workflow.add_node("analyst", analyst_node)
    print("       ✓ analyst")
    
    workflow.add_node("researcher", researcher_node)
    print("       ✓ researcher")
    
    workflow.add_node("report_writer", report_writer_node)
    print("       ✓ report_writer")
    
    # Step 3: Define the edges (workflow flow)
    print("   3️⃣  Adding edges:")
    
    # Linear flow: data_collector -> analyst
    workflow.add_edge("data_collector", "analyst")
    print("       ✓ data_collector → analyst")
    
    # CONDITIONAL EDGE: analyst -> (researcher OR report_writer)
    # This is where the magic happens!
    workflow.add_conditional_edges(
        "analyst",              # From this node
        should_research,        # Use this function to decide
        {
            "researcher": "researcher",      # If returns "researcher", go here
            "report_writer": "report_writer" # If returns "report_writer", go here
        }
    )
    print("       ✓ analyst → [CONDITIONAL] → researcher OR report_writer")
    
    # After research, always go to report writer
    workflow.add_edge("researcher", "report_writer")
    print("       ✓ researcher → report_writer")
    
    # Report writer is the end
    workflow.add_edge("report_writer", END)
    print("       ✓ report_writer → END")
    
    # Step 4: Set the entry point
    print("   4️⃣  Setting entry point: data_collector")
    workflow.set_entry_point("data_collector")
    
    # Step 5: Compile the graph
    print("   5️⃣  Compiling workflow...")
    compiled_workflow = workflow.compile()
    print("   ✅ Workflow compiled successfully!")
    
    return compiled_workflow

# ==============================================================================
# SECTION 10: MAIN EXECUTION
# ==============================================================================

def run_financial_research(ticker: str) -> dict:
    """
    Execute the full financial research workflow for a given stock.
    
    EDUCATIONAL NOTE:
    -----------------
    This is the main entry point for running our LangGraph workflow.
    
    WORKFLOW EXECUTION:
    1. Build the graph
    2. Create initial state
    3. Invoke the graph with state
    4. Receive final state with all results
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        
    Returns:
        Final state dictionary with complete research results
    """
    
    print_header(f"FINANCIAL RESEARCH WORKFLOW: {ticker}")
    print(f"   📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🎯 Target: {ticker}")
    
    # Build the workflow graph
    workflow = build_research_workflow()
    
    # Initialize the starting state
    # EDUCATIONAL NOTE: We provide initial values for ALL state fields
    initial_state: FinancialResearchState = {
        "ticker": ticker,
        "stock_data": {},
        "analysis": "",
        "needs_more_research": False,
        "research_notes": "",
        "final_report": "",
        "current_step": "initialized",
        "messages": []
    }
    
    print_header("EXECUTING WORKFLOW")
    print("   Starting workflow execution...")
    
    # Execute the workflow
    # The invoke() method runs the graph from start to END
    final_state = workflow.invoke(initial_state)
    
    print_header("WORKFLOW COMPLETE")
    print(f"   ✅ Final step: {final_state['current_step']}")
    print(f"   📊 Nodes executed: {len(final_state['messages'])}")
    
    return final_state

def main():
    """
    Main function demonstrating the LangGraph workflow.
    
    EDUCATIONAL NOTE:
    -----------------
    This demo shows how to:
    1. Validate the environment
    2. Run a research workflow
    3. Display the results
    
    Try changing the ticker to research different stocks!
    """
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  LANGGRAPH MULTI-AGENT FINANCIAL RESEARCH DEMO  ".center(68) + "█")
    print("█" + "  Advanced Level - FinResearch AI Team  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Step 1: Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Step 2: Get ticker from command line or use default
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "NVDA"  # Default: NVIDIA - interesting stock for analysis
    
    print_header("DEMO CONFIGURATION")
    print(f"   📈 Researching: {ticker}")
    print(f"   🔧 Framework: LangGraph")
    print(f"   🤖 Model: GPT-4o-mini")
    print("\n   💡 TIP: Run with a different ticker:")
    print(f"      python {os.path.basename(__file__)} AAPL")
    
    # Step 3: Run the research workflow
    try:
        final_state = run_financial_research(ticker)
        
        # Step 4: Display the final report
        print_header("FINAL RESEARCH REPORT")
        print(final_state["final_report"])
        
        # Show workflow statistics
        print_header("WORKFLOW STATISTICS")
        print(f"   📊 Ticker Analyzed: {final_state['ticker']}")
        print(f"   💰 Current Price: ${final_state['stock_data'].get('current_price', 'N/A')}")
        print(f"   🔬 Additional Research: {'Yes' if final_state['research_notes'] else 'No'}")
        print(f"   📝 Total Messages: {len(final_state['messages'])}")
        
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE - Thank you for exploring LangGraph!")
    print("=" * 70 + "\n")

# ==============================================================================
# SECTION 11: ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    EDUCATIONAL NOTE:
    -----------------
    The `if __name__ == "__main__"` pattern allows this file to be:
    1. Run directly: python demo_langgraph_workflow.py
    2. Imported as a module: from demo_langgraph_workflow import run_financial_research
    
    This makes the code reusable and testable!
    """
    main()
