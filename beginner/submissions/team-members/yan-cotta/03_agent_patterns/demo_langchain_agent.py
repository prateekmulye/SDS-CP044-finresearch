#!/usr/bin/env python3
"""
================================================================================
LANGCHAIN AGENT DEMO - FINANCIAL RESEARCH ASSISTANT
================================================================================

Welcome to the LangChain Agent Demo! This script demonstrates how to build
an intelligent financial research assistant using LangChain's agent framework.

🎯 LEARNING OBJECTIVES:
    1. Understand LangChain's tool-based architecture
    2. Learn how to convert Python functions into agent tools
    3. See how agents reason and select tools dynamically
    4. Experience interactive AI-powered financial research

📚 WHAT IS LANGCHAIN?
    LangChain is a framework for building applications with Large Language Models.
    It provides abstractions for:
    - Tools: Functions the LLM can call
    - Agents: LLMs that can reason and act
    - Chains: Sequences of operations
    - Memory: Conversation history management

🔧 TOOLS IN THIS DEMO:
    1. get_stock_price - Fetches current stock prices from Yahoo Finance
    2. get_company_overview - Gets company information and key metrics
    3. get_stock_history - Retrieves historical price data

💡 AGENT ARCHITECTURE:
    The agent uses a "tool calling" pattern where:
    1. User asks a question
    2. LLM decides which tool(s) to use
    3. Tool executes and returns data
    4. LLM synthesizes a response

Author: FinResearch AI Team (Yan Cotta)
Level: BEGINNER - Introduction to agentic AI with LangChain
================================================================================
"""

import os
import sys
from datetime import datetime

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# CONCEPT: We load API keys from a .env file to keep them secure.
# The .env file should be in the project root and NOT committed to git.
# This is a security best practice - NEVER hardcode API keys!

from dotenv import load_dotenv

def validate_environment() -> bool:
    """
    Find and load the .env file, then validate that required keys exist.
    
    EDUCATIONAL NOTE:
    -----------------
    This function demonstrates defensive programming. Instead of assuming
    the environment is correctly set up, we:
    1. Search for .env file in parent directories
    2. Validate that required keys are present
    3. Provide clear error messages if something is wrong
    
    Returns:
        True if environment is valid, False otherwise
    """
    # Start from current directory and search upward for .env
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search up to 5 directory levels for .env file
    for _ in range(5):
        env_path = os.path.join(current_dir, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✅ Loaded environment from: {env_path}")
            break
        current_dir = os.path.dirname(current_dir)
    else:
        print("⚠️  No .env file found, using system environment variables")
    
    # Check for required API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("\n📝 TO FIX THIS:")
        print("   1. Create a .env file in the project root directory")
        print("   2. Add this line: OPENAI_API_KEY=your-key-here")
        print("   3. Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    return True

# Validate environment before importing heavy dependencies
if not validate_environment():
    print("\n⛔ Cannot proceed without proper environment configuration.")
    sys.exit(1)

# =============================================================================
# STEP 1: IMPORT LANGCHAIN COMPONENTS
# =============================================================================
# CONCEPT: LangChain is modular - you import specific components you need.
# This keeps your application lightweight and explicit about dependencies.
#
# KEY IMPORTS EXPLAINED:
# - ChatOpenAI: Wrapper around OpenAI's chat models (handles API calls)
# - tool decorator: Converts Python functions into LangChain tools
# - create_react_agent: Factory function from LangGraph for ReAct-style agents
# - HumanMessage: Message type for user inputs

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# For our financial data tools
import yfinance as yf

# =============================================================================
# STEP 2: DEFINE TOOLS USING LANGCHAIN'S @tool DECORATOR
# =============================================================================
# CONCEPT: LangChain tools are Python functions decorated with @tool.
# The decorator automatically:
# - Extracts the function name as the tool name
# - Uses the docstring as the tool description (CRITICAL for agent reasoning)
# - Infers input types from type hints
#
# BEST PRACTICE: Write comprehensive docstrings! The agent reads these to
# decide when to use each tool. Include examples and edge cases.


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.
    
    Use this tool when you need to find out the current trading price of a stock.
    The input should be a stock ticker symbol like 'AAPL' for Apple, 'TSLA' for
    Tesla, 'GOOGL' for Google, or 'MSFT' for Microsoft.
    
    This tool returns the current price in USD along with the company name.
    
    Args:
        ticker: A stock ticker symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
        
    Returns:
        A string containing the company name and current stock price in USD
    """
    try:
        # Normalize input (handle lowercase, extra spaces)
        ticker = ticker.strip().upper()
        
        # Fetch stock data using yfinance
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        
        if history.empty:
            return f"Could not find data for ticker '{ticker}'. Please check the symbol."
        
        # Get the closing price
        current_price = history['Close'].iloc[-1]
        
        # Get company name for better readability
        info = stock.info
        company_name = info.get('shortName', ticker)
        
        return f"The current price of {company_name} ({ticker}) is ${current_price:.2f} USD"
        
    except Exception as e:
        return f"Error fetching stock price for '{ticker}': {str(e)}"


@tool
def get_company_overview(ticker: str) -> str:
    """
    Get a comprehensive overview of a company including sector, industry,
    market capitalization, and financial metrics.
    
    Use this tool when you need background information about a company,
    want to understand what business they're in, or need financial metrics
    like market cap, P/E ratio, and 52-week price range.
    
    This is useful for answering questions like:
    - "What does Tesla do?"
    - "What sector is Apple in?"
    - "What's Microsoft's market cap?"
    - "Give me an overview of NVIDIA"
    
    Args:
        ticker: A stock ticker symbol (e.g., 'AAPL', 'NVDA')
        
    Returns:
        A formatted string with company details and key financial metrics
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key information with sensible defaults
        company_name = info.get('shortName', 'Unknown')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 'N/A')
        week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
        week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
        summary = info.get('longBusinessSummary', 'No description available.')
        
        # Format market cap for readability
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f} Trillion"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f} Billion"
            else:
                market_cap_str = f"${market_cap/1e6:.2f} Million"
        else:
            market_cap_str = "N/A"
        
        # Truncate summary if too long
        if len(summary) > 400:
            summary = summary[:400] + "..."
        
        return f"""
Company: {company_name} ({ticker})
Sector: {sector}
Industry: {industry}
Market Cap: {market_cap_str}
P/E Ratio: {pe_ratio}
52-Week High: ${week_52_high}
52-Week Low: ${week_52_low}

Business Summary:
{summary}
"""
        
    except Exception as e:
        return f"Error fetching company info for '{ticker}': {str(e)}"


@tool
def get_stock_history(ticker: str, period: str = "1mo") -> str:
    """
    Get historical stock price data for technical analysis.
    
    Use this tool when you need to analyze price trends, understand how a
    stock has performed over time, or calculate returns.
    
    Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    Args:
        ticker: A stock ticker symbol
        period: Time period for historical data (default: '1mo')
        
    Returns:
        A summary of price movements and key statistics
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        if history.empty:
            return f"No historical data found for '{ticker}'"
        
        # Calculate key statistics
        start_price = history['Close'].iloc[0]
        end_price = history['Close'].iloc[-1]
        high_price = history['High'].max()
        low_price = history['Low'].min()
        avg_volume = history['Volume'].mean()
        
        # Calculate return
        price_change = end_price - start_price
        pct_change = (price_change / start_price) * 100
        
        return f"""
Historical Data for {ticker} (Period: {period})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Starting Price: ${start_price:.2f}
Current Price: ${end_price:.2f}
Price Change: ${price_change:.2f} ({pct_change:+.2f}%)
Period High: ${high_price:.2f}
Period Low: ${low_price:.2f}
Average Daily Volume: {avg_volume:,.0f}
"""
        
    except Exception as e:
        return f"Error fetching history for '{ticker}': {str(e)}"


# =============================================================================
# STEP 3: INITIALIZE THE LANGUAGE MODEL
# =============================================================================
# CONCEPT: ChatOpenAI is LangChain's wrapper around OpenAI's chat models.
# It provides a consistent interface that works with LangChain's other
# components (agents, chains, etc.).
#
# KEY PARAMETERS:
# - model: The OpenAI model to use (gpt-4o-mini is fast and cost-effective)
# - temperature: Controls randomness (0 = deterministic, 1 = creative)

print("\n🔧 Initializing LangChain components...")

llm = ChatOpenAI(
    model="gpt-4o-mini",  # Modern, cost-effective model with great tool calling
    temperature=0,  # We want consistent, factual responses for financial data
)

# Collect all our tools into a list
tools = [get_stock_price, get_company_overview, get_stock_history]

# =============================================================================
# STEP 4: CREATE THE REACT AGENT WITH LANGGRAPH
# =============================================================================
# CONCEPT: LangGraph provides a modern, graph-based agent architecture.
# The create_react_agent function creates a ReAct-style agent that:
# - Reasons about what tool to use (Thought)
# - Takes an action by calling a tool (Action)
# - Observes the result (Observation)
# - Repeats until it has a final answer
#
# SYSTEM PROMPT: Defines the agent's personality and behavior.
# This is critical for getting good responses!

SYSTEM_PROMPT = """You are a helpful financial research assistant with access to real-time stock market data.

Your capabilities:
- Look up current stock prices using the get_stock_price tool
- Provide company overviews and key metrics using the get_company_overview tool
- Analyze historical price data using the get_stock_history tool

When answering questions:
1. Use the available tools to get accurate, up-to-date information
2. Explain your findings clearly and concisely
3. Always mention that stock prices fluctuate and this is not investment advice

If you can't find information for a specific stock, let the user know politely."""

# Create the ReAct agent using LangGraph's prebuilt function
# This combines the LLM + Tools into a runnable agent graph
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

# =============================================================================
# STEP 5: AGENT IS READY!
# =============================================================================
# CONCEPT: With LangGraph's create_react_agent, the agent is already a
# complete runnable graph. No separate AgentExecutor needed!
#
# The agent:
# - Automatically handles the ReAct loop (Thought → Action → Observation)
# - Manages tool calling and response parsing
# - Returns the final answer when complete
#
# We invoke it by passing a dictionary with "messages" containing the user input.

print("✅ Agent initialized and ready!")
print("   Using LangGraph's ReAct agent with 3 financial tools")

# =============================================================================
# STEP 6: INTERACTIVE DEMO LOOP
# =============================================================================
# CONCEPT: This creates an interactive session where users can ask
# multiple questions. The agent maintains context within the session.

def print_header():
    """Print a nice header for the demo."""
    print("\n" + "=" * 70)
    print("  🤖 LANGCHAIN FINANCIAL RESEARCH AGENT")
    print("  An AI assistant with real-time stock market tools")
    print("=" * 70)
    print("\n📊 Available Tools:")
    print("   • get_stock_price - Current stock prices")
    print("   • get_company_overview - Company information & metrics")
    print("   • get_stock_history - Historical price data")
    print("\n💡 Example Questions:")
    print("   • What's the current price of Apple stock?")
    print("   • Tell me about Tesla as a company")
    print("   • How has NVIDIA performed over the last month?")
    print("   • Compare the prices of AAPL and MSFT")
    print("\n" + "=" * 70)
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 70 + "\n")


def run_interactive_session():
    """Run an interactive Q&A session with the agent."""
    print_header()
    
    # Store chat history for context
    chat_history = []
    
    while True:
        # Get user input
        try:
            user_input = input("\n🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Session ended. Thanks for using the demo!")
            break
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
            print("\n👋 Thanks for using the Financial Research Agent! Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            print("   (Please enter a question or type 'quit' to exit)")
            continue
        
        print("\n🤔 Agent is thinking...\n")
        print("-" * 50)
        
        try:
            # Invoke the agent with the user's input
            # LANGGRAPH PATTERN: The agent expects a dict with "messages"
            # The agent will:
            # 1. Analyze the question
            # 2. Decide which tools to use
            # 3. Call the tools
            # 4. Synthesize a response
            
            # Build the messages list with chat history
            messages = []
            for role, content in chat_history:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=user_input))
            
            # Invoke the agent
            response = agent.invoke({"messages": messages})
            
            # Extract the final response from the agent's output
            # The response contains all messages, we want the last AI message
            final_message = response["messages"][-1]
            output_text = final_message.content
            
            print("-" * 50)
            print(f"\n🤖 Agent: {output_text}")
            
            # Update chat history for context in future questions
            chat_history.extend([
                ("human", user_input),
                ("ai", output_text)
            ])
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("   Please try rephrasing your question.")


# =============================================================================
# STEP 7: MAIN ENTRY POINT
# =============================================================================
# CONCEPT: The if __name__ == "__main__" pattern allows this file to be:
# - Run directly (python demo_langchain_agent.py)
# - Imported as a module without running the demo

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  LANGCHAIN AGENT DEMO - Financial Research Assistant  ".center(68) + "█")
    print("█" + f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    run_interactive_session()
    
    print("\n" + "=" * 70)
    print("  Demo complete. You've learned how LangChain agents work!")
    print("=" * 70 + "\n")
