#!/usr/bin/env python3
"""
================================================================================
FINRESEARCH AI - WEEK 2: CREWAI MULTI-AGENT SYSTEM
================================================================================

LEARNING OBJECTIVES:
    1. Understand WHY we use multiple specialized agents instead of one "super agent"
    2. Learn how to assign specific tools to specific agents
    3. See how role separation reduces hallucination and improves accuracy
    4. Practice defining tasks that leverage agent specializations

THE TWO AGENTS WE'RE BUILDING:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                        RESEARCHER AGENT                           │
    │  ─────────────────────────────────────────────────────────────────  │
    │  ROLE:     Qualitative Data Specialist                              │
    │  FOCUS:    News, sentiment, market narratives                       │
    │  TOOLS:    SearchTool (DuckDuckGo)                                 │
    │  TEMP:     0.7 (allows creative summarization)                      │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    FINANCIAL ANALYST AGENT                        │
    │  ─────────────────────────────────────────────────────────────────  │
    │  ROLE:     Quantitative Data Specialist                             │
    │  FOCUS:    Prices, ratios, metrics, fundamentals                    │
    │  TOOLS:    FinancialDataTool (yfinance)                            │
    │  TEMP:     0.0 (precision required for numbers!)                    │
    └─────────────────────────────────────────────────────────────────────┘

TEACHING NOTE - WHY NOT ONE "SUPER AGENT"?
   
   You might think: "Why not just give ONE agent ALL the tools?"
   
   Here's why that's problematic:
   
   1. CONTEXT OVERLOAD
      - LLMs have limited context windows
      - More tools = more confusion about which to use
      - Specialized agents have focused, clear responsibilities
   
   2. HALLUCINATION RISK
      - When an agent has too many capabilities, it might "imagine"
        it can do things it can't, or mix up tool outputs
      - A researcher won't accidentally "calculate" P/E ratios
      - An analyst won't make up news articles
   
   3. PROMPT ENGINEERING
      - Specialized prompts work better than generic ones
      - "You are a financial analyst who focuses on numbers..." works
        better than "You can do everything..."
   
   4. TESTABILITY
      - Easier to test and debug individual agents
      - If news search fails, you know it's the Researcher
      - If price data is wrong, you know it's the Analyst

DEPENDENCIES:
    pip install crewai crewai-tools langchain-openai yfinance duckduckgo-search

Author: Yan Cotta | FinResearch AI Project
Week: 2 - Agent Architecture & Tool Assignment
================================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# =============================================================================
# STEP 0: ENVIRONMENT SETUP
# =============================================================================
# TEACHING NOTE: CrewAI requires an OpenAI API key (or another LLM provider).
# We validate this BEFORE importing heavy dependencies to fail fast with
# a helpful message.

from dotenv import load_dotenv

# Find .env file in project root
project_root = Path(__file__).resolve().parents[4]
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Environment loaded from: {env_path}")
else:
    load_dotenv()  # Try current directory

# Validate API key
def validate_environment():
    """Ensure OpenAI API key is set before proceeding."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 70)
        print("ERROR: OPENAI_API_KEY not set!")
        print("=" * 70)
        print("\nCrewAI requires an OpenAI API key (or alternative LLM).")
        print("\nTo fix:")
        print("  1. Create a .env file in the project root")
        print("  2. Add: OPENAI_API_KEY=sk-your-key-here")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    if not api_key.startswith("sk-"):
        print("WARNING: API key doesn't start with 'sk-' - may be invalid")
    
    return True

validate_environment()


# =============================================================================
# STEP 1: IMPORT CREWAI COMPONENTS
# =============================================================================
# TEACHING NOTE: CrewAI has three core concepts:
#   - Agent: The "who" (a persona with role, goal, backstory)
#   - Task: The "what" (a specific assignment)
#   - Crew: The "orchestrator" (manages agents and workflow)

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI

# For our custom tools
import yfinance as yf
from duckduckgo_search import DDGS
from pydantic import Field


# =============================================================================
# STEP 2: DEFINE CUSTOM TOOLS
# =============================================================================
# TEACHING NOTE - TOOL DESIGN PRINCIPLES:
#   1. Single Responsibility: Each tool does ONE thing
#   2. Clear Description: The description tells the agent WHEN to use it
#   3. Robust Error Handling: Tools should never crash the agent
#   4. Clean Output: Return strings the LLM can easily understand

# -----------------------------------------------------------------------------
# TOOL 2.1: FINANCIAL DATA TOOL (For the Analyst Agent)
# -----------------------------------------------------------------------------
# TEACHING NOTE: We extend BaseTool from CrewAI. This requires:
#   - name: Tool identifier (used in logs)
#   - description: CRITICAL - the agent reads this to decide when to use it
#   - _run(): The actual implementation

class FinancialDataTool(BaseTool):
    """
    Tool for fetching quantitative financial data from Yahoo Finance.
    
    TEACHING NOTE - WHY A CLASS?
        CrewAI tools can be functions or classes. We use a class here because:
        1. It's more explicit about the tool's interface
        2. Easier to add configuration (like API keys) later
        3. Better for complex tools with helper methods
    """
    
    name: str = "financial_data_tool"
    description: str = (
        "Fetches quantitative financial data for a stock ticker. "
        "Use this tool to get: current stock price, market cap, P/E ratio, "
        "EPS, beta, 52-week high/low, volume, and other financial metrics. "
        "Input should be a stock ticker symbol like 'AAPL', 'TSLA', or 'GOOGL'. "
        "This tool returns NUMBERS and FACTS, not news or opinions."
    )
    
    def _run(self, ticker: str) -> str:
        """
        Fetches financial data for the given ticker.
        
        TEACHING NOTE: The _run method is what gets executed when the agent
        calls this tool. It MUST return a string (not a dict or object).
        The LLM will parse this string to extract information.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            A formatted string with financial data
        """
        try:
            # Normalize input
            ticker = ticker.strip().upper()
            
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Validate we got data
            if not info or not info.get('regularMarketPrice'):
                # Try to get any price data
                history = stock.history(period="1d")
                if history.empty:
                    return f"ERROR: Could not find data for ticker '{ticker}'. Please verify the symbol is correct."
            
            # Extract key metrics
            current_price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
            
            # Format market cap
            market_cap = info.get('marketCap', 0)
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f} Trillion"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f} Billion"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f} Million"
            else:
                market_cap_str = "N/A"
            
            # Build response
            # TEACHING NOTE: We structure this clearly so the LLM can parse it
            return f"""
FINANCIAL DATA FOR {info.get('shortName', ticker)} ({ticker})
{'='*50}

PRICE INFORMATION:
- Current Price: ${current_price:.2f} {info.get('currency', 'USD')}
- Day High: ${info.get('dayHigh', 'N/A')}
- Day Low: ${info.get('dayLow', 'N/A')}
- 52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
- 52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}

VALUATION METRICS:
- Market Cap: {market_cap_str}
- P/E Ratio (TTM): {info.get('trailingPE', 'N/A')}
- Forward P/E: {info.get('forwardPE', 'N/A')}
- EPS (TTM): ${info.get('trailingEps', 'N/A')}
- Price-to-Book: {info.get('priceToBook', 'N/A')}

TRADING INFORMATION:
- Volume: {info.get('volume', 'N/A'):,}
- Average Volume: {info.get('averageVolume', 'N/A'):,}
- Beta: {info.get('beta', 'N/A')}

COMPANY INFO:
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}

Data fetched at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
        except Exception as e:
            return f"ERROR fetching data for '{ticker}': {type(e).__name__}: {str(e)}"


# -----------------------------------------------------------------------------
# TOOL 2.2: NEWS SEARCH TOOL (For the Researcher Agent)
# -----------------------------------------------------------------------------

class NewsSearchTool(BaseTool):
    """
    Tool for searching recent news and sentiment about companies.
    
    TEACHING NOTE - QUALITATIVE VS QUANTITATIVE:
        This tool is for QUALITATIVE research:
        - What are people saying about the company?
        - What recent events have occurred?
        - What's the market sentiment?
        
        It does NOT provide numbers or financial metrics.
        That's the Analyst's job with the FinancialDataTool.
    """
    
    name: str = "news_search_tool"
    description: str = (
        "Searches for recent news articles, press releases, and market sentiment "
        "about a company or stock. Use this tool to find: recent company news, "
        "earnings announcements, product launches, executive changes, analyst opinions, "
        "and market sentiment. Input should be a company name or topic like "
        "'Apple earnings' or 'Tesla latest news'. This tool returns NEWS and OPINIONS, "
        "not financial numbers or metrics."
    )
    
    def _run(self, query: str) -> str:
        """
        Searches for news about the given topic.
        
        Args:
            query: Search query (company name, topic, etc.)
            
        Returns:
            A formatted string with news summaries
        """
        try:
            # Clean query
            query = query.strip()
            
            if not query:
                return "ERROR: Empty search query. Please provide a company name or topic."
            
            # Add financial context if not present
            financial_keywords = ['stock', 'share', 'market', 'earnings', 'investor', 'news']
            if not any(kw in query.lower() for kw in financial_keywords):
                query = f"{query} stock market news"
            
            # Search using DuckDuckGo
            articles = []
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    keywords=query,
                    max_results=5
                ))
            
            if not results:
                return f"No recent news found for: {query}"
            
            # Format results
            # TEACHING NOTE: We number the articles and include sources
            # so the LLM can reference specific articles in its analysis
            output_lines = [
                f"NEWS SEARCH RESULTS FOR: {query}",
                "=" * 50,
                ""
            ]
            
            for i, article in enumerate(results, 1):
                output_lines.append(f"[Article {i}]")
                output_lines.append(f"Title: {article.get('title', 'No title')}")
                output_lines.append(f"Source: {article.get('source', 'Unknown')}")
                output_lines.append(f"Date: {article.get('date', 'Unknown')}")
                output_lines.append(f"Summary: {article.get('body', 'No summary available.')}")
                output_lines.append("")
            
            output_lines.append(f"Search completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"ERROR searching news for '{query}': {type(e).__name__}: {str(e)}"


# =============================================================================
# STEP 3: DEFINE THE AGENTS
# =============================================================================
# TEACHING NOTE - AGENT DESIGN PHILOSOPHY:
#   Each agent is like hiring a specialist for your team:
#   
#   - ROLE: Their job title (shapes how they approach problems)
#   - GOAL: Their motivation (what success looks like to them)
#   - BACKSTORY: Their personality (affects communication style)
#   - TOOLS: Their toolkit (strictly what they CAN access)
#   
#   Notice how we DON'T give all tools to all agents!

# -----------------------------------------------------------------------------
# AGENT 3.1: THE RESEARCHER
# -----------------------------------------------------------------------------

def create_researcher_agent() -> Agent:
    """
    Creates the Researcher Agent specialized in qualitative analysis.
    
    TEACHING NOTE - TEMPERATURE SETTING:
        We set temperature=0.7 for the Researcher because:
        - Summarizing news requires some creativity
        - Identifying sentiment involves interpretation
        - We want varied, natural-sounding summaries
        
        We DON'T want temperature=0 here because that would make
        summaries repetitive and robotic.
    """
    
    return Agent(
        role='Senior Market Research Analyst',
        
        goal=(
            'Find and synthesize the latest news, sentiment, and qualitative '
            'information about companies. Identify key narratives, market '
            'perceptions, and important recent developments that could impact '
            'investment decisions.'
        ),
        
        backstory=(
            "You are a seasoned market research analyst with 15 years of experience "
            "at top investment firms. You have a keen eye for separating signal from "
            "noise in financial news. You understand that market perception often "
            "matters as much as fundamentals. You're skilled at identifying emerging "
            "trends and sentiment shifts before they become obvious to everyone. "
            "You always cite your sources and distinguish between facts, opinions, "
            "and speculation."
        ),
        
        # TEACHING NOTE: This agent ONLY gets the news search tool
        # It cannot access financial data - that's the Analyst's job!
        tools=[NewsSearchTool()],
        
        # TEACHING NOTE: Higher temperature for creative summarization
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7),
        
        verbose=True,  # Show reasoning during development
        allow_delegation=False,  # Must do its own work
        
        # Memory helps maintain context across interactions
        memory=True,
    )


# -----------------------------------------------------------------------------
# AGENT 3.2: THE FINANCIAL ANALYST
# -----------------------------------------------------------------------------

def create_analyst_agent() -> Agent:
    """
    Creates the Financial Analyst Agent specialized in quantitative analysis.
    
    TEACHING NOTE - TEMPERATURE SETTING:
        We set temperature=0 for the Analyst because:
        - Financial calculations require PRECISION
        - P/E ratios, prices, etc. must be EXACT
        - We don't want the model to "be creative" with numbers!
        
        A temperature of 0 makes the model deterministic and focused
        on accuracy over creativity.
    """
    
    return Agent(
        role='Senior Financial Analyst',
        
        goal=(
            'Analyze quantitative financial data including stock prices, '
            'valuation ratios, market metrics, and fundamental indicators. '
            'Provide precise, data-driven insights for investment analysis.'
        ),
        
        backstory=(
            "You are a CFA charterholder with a decade of experience in equity "
            "research. You have a strong background in financial modeling and "
            "valuation. You believe in letting the numbers tell the story and "
            "are meticulous about accuracy. You never guess at financial figures - "
            "if you don't have the data, you say so. You're skilled at contextualizing "
            "metrics (e.g., 'a P/E of 25 is high for this industry') and identifying "
            "red flags in financial data."
        ),
        
        # TEACHING NOTE: This agent ONLY gets the financial data tool
        # It cannot search news - that's the Researcher's job!
        tools=[FinancialDataTool()],
        
        # TEACHING NOTE: Zero temperature for numerical precision
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        
        verbose=True,
        allow_delegation=False,
        memory=True,
    )


# =============================================================================
# STEP 4: DEFINE SAMPLE TASKS
# =============================================================================
# TEACHING NOTE - TASK STRUCTURE:
#   Tasks define the WHAT (not the HOW). The agent decides how to accomplish
#   the task based on its role, goal, and available tools.
#   
#   Key fields:
#   - description: What needs to be done (be specific!)
#   - expected_output: What format should the result be in
#   - agent: Who is responsible (determines which tools are available)

def create_research_task(ticker: str, researcher: Agent) -> Task:
    """
    Creates a research task for the Researcher Agent.
    
    TEACHING NOTE: Notice how the task description focuses on QUALITATIVE
    aspects. We're not asking for P/E ratios or prices - that's not
    what the Researcher does.
    """
    
    return Task(
        description=(
            f"Research the company with ticker symbol {ticker}. "
            f"Find recent news, market sentiment, and any notable developments. "
            f"Focus on:\n"
            f"1. Recent news headlines (last 1-2 weeks)\n"
            f"2. Any earnings announcements or guidance\n"
            f"3. Product launches or strategic initiatives\n"
            f"4. Analyst opinions or rating changes\n"
            f"5. Overall market sentiment (bullish/bearish/neutral)\n\n"
            f"Synthesize your findings into a coherent narrative."
        ),
        
        expected_output=(
            "A structured research summary with:\n"
            "- Executive Summary (2-3 sentences)\n"
            "- Key News Items (bulleted list with sources)\n"
            "- Sentiment Assessment (bullish/bearish/neutral with reasoning)\n"
            "- Notable Risks or Opportunities identified"
        ),
        
        agent=researcher
    )


def create_analysis_task(ticker: str, analyst: Agent) -> Task:
    """
    Creates an analysis task for the Financial Analyst Agent.
    
    TEACHING NOTE: This task focuses purely on QUANTITATIVE aspects.
    We're asking for numbers, metrics, and data-driven insights.
    """
    
    return Task(
        description=(
            f"Analyze the financial data for {ticker}. "
            f"Retrieve current market data and provide a quantitative assessment. "
            f"Focus on:\n"
            f"1. Current stock price and recent price movement\n"
            f"2. Valuation metrics (P/E, Price-to-Book, etc.)\n"
            f"3. Market capitalization and trading volume\n"
            f"4. 52-week price range context\n"
            f"5. Any notable metrics that stand out (high/low relative to industry)\n\n"
            f"Provide precise numbers and contextualize what they mean."
        ),
        
        expected_output=(
            "A structured financial analysis with:\n"
            "- Price Summary (current price, 52-week position)\n"
            "- Valuation Metrics (P/E, P/B, with interpretation)\n"
            "- Key Observations (what stands out in the data)\n"
            "- Data Quality Notes (any metrics that couldn't be retrieved)"
        ),
        
        agent=analyst
    )


# =============================================================================
# STEP 5: CREATE AND RUN THE CREW
# =============================================================================
# TEACHING NOTE - CREW CONFIGURATION:
#   The Crew orchestrates the agents. Key settings:
#   
#   - process=Process.sequential: Tasks run one after another
#   - verbose=True: See the step-by-step execution
#   - memory=True: Crew remembers context across tasks

def run_research_crew(ticker: str):
    """
    Creates and executes a research crew for the given ticker.
    
    This demonstrates:
    1. Creating specialized agents
    2. Assigning appropriate tasks
    3. Orchestrating execution
    """
    
    print("\n" + "=" * 70)
    print(f"  LAUNCHING FINRESEARCH AI CREW FOR: {ticker}")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create our specialized agents
    print("\nCreating agents...")
    researcher = create_researcher_agent()
    analyst = create_analyst_agent()
    print("  [OK] Researcher Agent (qualitative focus)")
    print("  [OK] Analyst Agent (quantitative focus)")
    
    # Create tasks for each agent
    print("\nCreating tasks...")
    research_task = create_research_task(ticker, researcher)
    analysis_task = create_analysis_task(ticker, analyst)
    print(f"  [OK] Research Task for {ticker}")
    print(f"  [OK] Analysis Task for {ticker}")
    
    # Assemble the crew
    print("\nAssembling crew...")
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        
        # Sequential process: Researcher first, then Analyst
        # TEACHING NOTE: In Week 3, we might use hierarchical process
        # where a "Manager" agent coordinates the specialists
        process=Process.sequential,
        
        verbose=True,
        memory=True,
    )
    
    # Execute!
    print("\n" + "=" * 70)
    print("  CREW EXECUTION STARTING...")
    print("  Watch the agents work below:")
    print("=" * 70 + "\n")
    
    try:
        result = crew.kickoff()
        
        print("\n" + "=" * 70)
        print("  [OK] CREW EXECUTION COMPLETE")
        print("=" * 70)
        print("\nFINAL OUTPUT:")
        print("-" * 70)
        print(result)
        print("-" * 70)
        
        return result
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"  [FAIL] CREW EXECUTION FAILED: {type(e).__name__}")
        print("=" * 70)
        print(f"\nError details: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your OPENAI_API_KEY is valid")
        print("  2. Ensure you have internet connectivity")
        print("  3. Try a different ticker symbol")
        return None


# =============================================================================
# STEP 6: DEMO MODE - TEST INDIVIDUAL AGENTS
# =============================================================================
# TEACHING NOTE: Before running the full crew, it's useful to test
# each agent individually. This helps with debugging.

def test_tools_directly():
    """
    Tests the tools directly without going through agents.
    Useful for debugging and verifying tool functionality.
    """
    
    print("\n" + "=" * 70)
    print("  TESTING TOOLS DIRECTLY")
    print("=" * 70)
    
    # Test Financial Data Tool
    print("\nTesting FinancialDataTool with 'AAPL'...")
    print("-" * 50)
    fin_tool = FinancialDataTool()
    fin_result = fin_tool._run("AAPL")
    print(fin_result)
    
    # Test News Search Tool
    print("\nTesting NewsSearchTool with 'Apple'...")
    print("-" * 50)
    news_tool = NewsSearchTool()
    news_result = news_tool._run("Apple Inc")
    print(news_result)
    
    print("\n" + "=" * 70)
    print("  [OK] TOOL TESTS COMPLETE")
    print("=" * 70)


def run_single_agent_demo(ticker: str = "MSFT"):
    """
    Demonstrates running a single agent with a single task.
    Good for testing individual agent behavior.
    """
    
    print("\n" + "=" * 70)
    print(f"  SINGLE AGENT DEMO: Analyst Agent for {ticker}")
    print("=" * 70)
    
    # Create just the analyst
    analyst = create_analyst_agent()
    task = create_analysis_task(ticker, analyst)
    
    # Create a minimal crew
    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    
    print("\n" + "=" * 70)
    print("  ANALYST OUTPUT:")
    print("=" * 70)
    print(result)
    
    return result


# =============================================================================
# STEP 7: MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point with menu for different demo modes.
    """
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  FINRESEARCH AI - WEEK 2: MULTI-AGENT DEMO  ".center(68) + "█")
    print("█" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\nTEACHING NOTES:")
    print("-" * 70)
    print("This demo showcases CrewAI's multi-agent architecture with:")
    print("  • RESEARCHER AGENT: Qualitative analysis (news, sentiment)")
    print("  • ANALYST AGENT: Quantitative analysis (prices, ratios)")
    print("")
    print("Key Concept: Each agent has DIFFERENT tools for DIFFERENT jobs!")
    print("-" * 70)
    
    print("\nDEMO OPTIONS:")
    print("  1. Test tools directly (no LLM calls)")
    print("  2. Run single agent (Analyst only)")
    print("  3. Run full crew (Researcher + Analyst)")
    print("  4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                test_tools_directly()
            elif choice == "2":
                ticker = input("Enter ticker symbol (default: MSFT): ").strip().upper() or "MSFT"
                run_single_agent_demo(ticker)
            elif choice == "3":
                ticker = input("Enter ticker symbol (default: AAPL): ").strip().upper() or "AAPL"
                run_research_crew(ticker)
            elif choice == "4":
                print("\nGoodbye! Happy researching!\n")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            print("Try again or choose a different option.")


if __name__ == "__main__":
    main()
