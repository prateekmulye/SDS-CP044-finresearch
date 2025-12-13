#!/usr/bin/env python3
"""
================================================================================
FinResearch AI - Advanced Track Demo: Multi-Agent Financial Crew
================================================================================

PURPOSE:
    This script demonstrates INTERMEDIATE CrewAI concepts:
    1. Custom Tools    - Giving agents the ability to fetch real-world data
    2. Multi-Agent     - Two specialized agents with different capabilities
    3. Sequential Flow - Output from one agent becomes input for another
    4. Context Passing - Explicit task dependencies using the 'context' parameter

LEARNING OBJECTIVES:
    - Understand how to create custom tools using the @tool decorator
    - Learn the difference between "tool-enabled" and "tool-less" agents
    - See how agents collaborate in a sequential workflow
    - Observe real API calls (Yahoo Finance) integrated into agent reasoning

ARCHITECTURE:
    ┌─────────────────┐         ┌─────────────────┐
    │ Market Researcher│ ───────▶│ Investment Writer│
    │ (has tools)      │  data   │ (no tools)       │
    └─────────────────┘         └─────────────────┘

BEFORE RUNNING:
    1. Ensure you have installed dependencies: pip install -r requirements.txt
    2. Set your OpenAI API key: export OPENAI_API_KEY="sk-..."
    
Author: Yan Cotta | FinResearch AI Project
================================================================================
"""

import os
import sys
from pathlib import Path

# =============================================================================
# STEP 0A: LOAD ENVIRONMENT VARIABLES FROM .env FILE
# =============================================================================
# WHY WE USE DOTENV: Storing API keys in code is a security anti-pattern.
# The .env file keeps secrets out of version control (it's in .gitignore).
# python-dotenv loads these variables into os.environ automatically.

from dotenv import load_dotenv

# Find the .env file in the project root (navigate up from this script's location)
# This approach works regardless of where you run the script from.
project_root = Path(__file__).resolve().parents[5]  # Go up 5 levels to reach project root
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    # Try current working directory as fallback
    load_dotenv()

# =============================================================================
# STEP 0B: ENVIRONMENT VALIDATION (Safety First!)
# =============================================================================
# WHY THIS MATTERS: API key errors in multi-agent systems are harder to debug
# because they may occur mid-execution. Validating upfront saves headaches.

def validate_environment():
    """
    Validates that required environment variables are set.
    Returns True if valid, exits with helpful message if not.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 70)
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nThis demo requires an OpenAI API key to function.")
        print("To fix this, run one of the following commands:\n")
        print("  Linux/Mac:  export OPENAI_API_KEY='sk-your-key-here'")
        print("  Windows:    set OPENAI_API_KEY=sk-your-key-here")
        print("\nOr create a .env file in the project root with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    if not api_key.startswith("sk-"):
        print("\n⚠️  WARNING: Your OPENAI_API_KEY doesn't start with 'sk-'.")
        print("   This may indicate an invalid key format.\n")
    
    return True

# Run validation before any imports that might use the API key
validate_environment()

# =============================================================================
# STEP 1: IMPORTS
# =============================================================================
# CONCEPT: We import yfinance for real market data. The @tool decorator from
# crewai.tools is the modern way to create tools (preferred over langchain.tools
# for CrewAI-specific projects as of crewai >= 0.30.0).

import yfinance as yf
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool  # UPDATED: Using crewai's native @tool decorator
from langchain_openai import ChatOpenAI

# =============================================================================
# STEP 2: DEFINE CUSTOM TOOLS
# =============================================================================
# CONCEPT: Tools are functions that agents can "call" during their reasoning.
# This is what makes agents truly powerful - they can interact with the real
# world, not just generate text based on training data.
#
# ANATOMY OF A TOOL:
#   - @tool decorator: Registers the function as a tool with a name
#   - Docstring: CRITICAL! This is what the LLM reads to decide when to use it
#   - Return value: Should be a string that the agent can incorporate into its response
#
# BEST PRACTICE: Write docstrings as if explaining to a coworker when they
# should use this function. Include input format examples!


@tool("Get Stock Price")
def get_stock_price(ticker: str) -> str:
    """
    Fetches the current stock price for a given ticker symbol.
    
    Use this tool when you need to find the current market price of a stock.
    Input should be a valid stock ticker symbol (e.g., 'AAPL' for Apple, 
    'TSLA' for Tesla, 'GOOGL' for Google).
    
    Args:
        ticker: A stock ticker symbol (uppercase recommended)
    
    Returns:
        A string with the current stock price or an error message
    """
    try:
        # DEFENSIVE CODING: Clean the input to handle common formatting issues
        ticker = ticker.strip().upper()
        
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        
        # EDGE CASE: Handle empty data (invalid ticker or market closed)
        if history.empty:
            return f"No data available for ticker '{ticker}'. Please verify the ticker symbol is correct."
        
        price = history['Close'].iloc[-1]
        
        # CONTEXT ENRICHMENT: Return additional useful information
        company_name = stock.info.get('shortName', ticker)
        return f"The current price of {company_name} ({ticker}) is ${price:.2f} USD"
        
    except Exception as e:
        # GRACEFUL DEGRADATION: Return useful error info instead of crashing
        return f"Error fetching price for '{ticker}': {str(e)}. Please check the ticker symbol."


@tool("Get Stock Info")
def get_stock_info(ticker: str) -> str:
    """
    Fetches detailed company information for a given ticker symbol.
    
    Use this tool when you need background information about a company,
    such as sector, industry, market cap, or business description.
    Input should be a valid stock ticker symbol (e.g., 'AAPL', 'MSFT').
    
    Args:
        ticker: A stock ticker symbol
    
    Returns:
        A string with company details or an error message
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # BUILD A STRUCTURED RESPONSE
        # We return key metrics that would be useful for a financial analyst
        return (
            f"Company: {info.get('shortName', 'N/A')}\n"
            f"Sector: {info.get('sector', 'N/A')}\n"
            f"Industry: {info.get('industry', 'N/A')}\n"
            f"Market Cap: ${info.get('marketCap', 0):,.0f}\n"
            f"52-Week High: ${info.get('fiftyTwoWeekHigh', 0):.2f}\n"
            f"52-Week Low: ${info.get('fiftyTwoWeekLow', 0):.2f}"
        )
    except Exception as e:
        return f"Error fetching info for '{ticker}': {str(e)}"


# =============================================================================
# STEP 3: DEFINE AGENTS
# =============================================================================
# CONCEPT: In a multi-agent system, each agent has a SPECIALIZED role.
# Think of it like a real team: you wouldn't ask your data scientist to
# write marketing copy, or your copywriter to run SQL queries.
#
# KEY INSIGHT: The researcher has TOOLS (can fetch data), while the writer
# has NO TOOLS (synthesizes information). This separation of concerns
# makes each agent more focused and effective.

# AGENT 1: THE RESEARCHER
# This agent's superpower is accessing real-time market data through tools.
market_researcher = Agent(
    role='Lead Market Researcher',
    goal='Gather accurate, real-time stock data and identify key market trends',
    backstory=(
        "You are a meticulous, data-driven market researcher with 10 years "
        "of experience at top investment firms. You never make claims without "
        "backing them up with concrete numbers. You're known for your ability "
        "to quickly gather and synthesize market data into actionable insights."
    ),
    verbose=True,
    
    # TOOLS: This is the key differentiator! This agent can fetch real data.
    # The agent will automatically decide when to use these tools based on
    # the task description and the tool docstrings.
    tools=[get_stock_price, get_stock_info],
    
    # ALLOW DELEGATION: Set to False because we want this agent to do the
    # research itself, not delegate to the writer (who has no tools anyway).
    allow_delegation=False,
    
    # LLM: We use GPT-4 with temperature=0 for maximum accuracy.
    # For financial data, we want deterministic, factual responses.
    llm=ChatOpenAI(model_name="gpt-4", temperature=0)
)

# AGENT 2: THE WRITER
# This agent's superpower is transforming dry data into compelling narratives.
investment_writer = Agent(
    role='Senior Investment Newsletter Writer',
    goal='Transform market research into engaging, accessible investment content',
    backstory=(
        "You are a world-renowned financial journalist who has written for "
        "The Wall Street Journal, Bloomberg, and Financial Times. You have a "
        "gift for making complex market dynamics understandable to everyday "
        "investors. You balance being informative with being engaging."
    ),
    verbose=True,
    
    # NO TOOLS: This agent works purely with information provided to it.
    # It doesn't need to fetch data - that's the researcher's job.
    tools=[],
    
    # DELEGATION: Also False - this is the last agent in the chain.
    allow_delegation=False,
    
    # LLM: We use temperature=0.7 for more creative, engaging writing
    # while still maintaining coherence and accuracy.
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.7)
)

# =============================================================================
# STEP 4: DEFINE TASKS
# =============================================================================
# CONCEPT: Tasks define WHAT needs to be done. In a sequential workflow,
# the output of one task automatically becomes available to subsequent tasks.
#
# KEY PATTERN: Research → Analysis → Synthesis → Output
# This mirrors how real financial teams work!

# TASK 1: RESEARCH
# The researcher will use their tools to gather real market data.
research_task = Task(
    description=(
        "Research the current market status of Tesla (TSLA) and Apple (AAPL). "
        "For each stock, find:\n"
        "1. The current stock price\n"
        "2. Basic company information\n"
        "Compile your findings into a structured research report."
    ),
    expected_output=(
        "A structured research report with two sections (one per stock), "
        "each containing the current price and key company metrics. "
        "Include the date/time context that this is real-time data."
    ),
    agent=market_researcher
)

# TASK 2: WRITING
# The writer will transform the research into a polished newsletter.
write_task = Task(
    description=(
        "Using the research data provided, write a compelling 'Market Flash' "
        "newsletter that compares Tesla and Apple. Your newsletter should:\n"
        "1. Have a catchy, attention-grabbing headline\n"
        "2. Summarize the key price points\n"
        "3. Provide brief context on what these prices might mean for investors\n"
        "4. End with a thought-provoking question or insight"
    ),
    expected_output=(
        "A 150-200 word newsletter in professional format with:\n"
        "- A compelling headline\n"
        "- Opening hook\n"
        "- Key data points (prices)\n"
        "- Brief analysis\n"
        "- Closing thought"
    ),
    agent=investment_writer,
    
    # CONTEXT: This explicitly tells CrewAI that this task depends on the
    # output from research_task. While CrewAI can infer this in sequential
    # mode, being explicit is a best practice for maintainability.
    context=[research_task]
)

# =============================================================================
# STEP 5: ASSEMBLE AND EXECUTE THE CREW
# =============================================================================
# CONCEPT: The Crew brings everything together. In sequential mode, it will:
#   1. Execute research_task with market_researcher
#   2. Pass the output to write_task
#   3. Execute write_task with investment_writer
#   4. Return the final newsletter

financial_crew = Crew(
    agents=[market_researcher, investment_writer],
    tasks=[research_task, write_task],
    
    # SEQUENTIAL PROCESS: Tasks execute in order, with outputs flowing forward.
    # This is perfect for pipelines where each step builds on the previous.
    process=Process.sequential,
    
    # VERBOSE: Enables detailed logging of the entire crew execution.
    verbose=True
)

# =============================================================================
# STEP 6: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 STARTING FINANCIAL RESEARCH CREW")
    print("=" * 70)
    print("\nThis demo will:")
    print("  1. Fetch real stock prices for TSLA and AAPL")
    print("  2. Research company information")
    print("  3. Generate a professional newsletter")
    print("\nWatch the verbose output to see each agent's reasoning...\n")
    print("=" * 70 + "\n")
    
    try:
        result = financial_crew.kickoff()
        
        print("\n" + "=" * 70)
        print("📰 FINAL NEWSLETTER")
        print("=" * 70)
        print(result)
        print("=" * 70)
        print("\n✅ Crew execution completed successfully!\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ EXECUTION ERROR: {type(e).__name__}")
        print("=" * 70)
        print(f"\nError Details: {e}")
        print("\nTroubleshooting Steps:")
        print("  1. Verify your OPENAI_API_KEY is valid and has available credits")
        print("  2. Check your internet connection (needed for Yahoo Finance API)")
        print("  3. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  4. If using GPT-4, verify your API key has GPT-4 access")
        print("\nFor GPT-4 access issues, you can modify the script to use 'gpt-3.5-turbo'")
        print("=" * 70 + "\n")
        sys.exit(1)