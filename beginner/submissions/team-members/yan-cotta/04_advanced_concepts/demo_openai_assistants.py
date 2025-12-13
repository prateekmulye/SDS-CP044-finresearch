#!/usr/bin/env python3
"""
================================================================================
FinResearch AI - Advanced Track Demo: OpenAI Assistants API
================================================================================

PURPOSE:
    This script demonstrates OpenAI's Assistants API - a more sophisticated
    approach to building AI agents compared to raw Chat Completions. The
    Assistants API provides:
    
    - Persistent Threads: Conversation history managed by OpenAI
    - Built-in Tools: Code Interpreter, File Search, Function Calling
    - Stateful Agents: Assistants persist across sessions
    - Managed Runs: Automatic handling of multi-step tool execution

ASSISTANTS vs CHAT COMPLETIONS:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Chat Completions (Basic)                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  • You manage conversation history                                      │
    │  • You handle tool execution loops manually                            │
    │  • Stateless - each call is independent                                │
    │  • More control, more code                                              │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Assistants API (Advanced)                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  • OpenAI manages conversation history (Threads)                        │
    │  • Automatic tool execution with polling                                │
    │  • Persistent Assistants with defined behaviors                         │
    │  • Built-in Code Interpreter and File Search                            │
    │  • Less code, managed complexity                                        │
    └─────────────────────────────────────────────────────────────────────────┘

KEY CONCEPTS COVERED:
    1. Assistant Creation - Defining a persistent AI assistant
    2. Threads - Managing conversation sessions
    3. Messages - Adding user messages to threads
    4. Runs - Executing the assistant on a thread
    5. Function Calling - Custom tools with the Assistants API
    6. Polling - Waiting for runs to complete

ARCHITECTURE:
    ┌──────────┐     ┌──────────────┐     ┌──────────────┐
    │  CLIENT  │────▶│   ASSISTANT  │────▶│    THREAD    │
    └──────────┘     └──────────────┘     └──────────────┘
         │                  │                     │
         │                  │                     │
         ▼                  ▼                     ▼
    ┌──────────┐     ┌──────────────┐     ┌──────────────┐
    │ Run Loop │◀───│    TOOLS     │◀───│   MESSAGES   │
    └──────────┘     └──────────────┘     └──────────────┘

BEFORE RUNNING:
    1. Install dependencies: pip install -r requirements.txt
    2. Ensure your .env file contains: OPENAI_API_KEY=sk-...

Author: Yan Cotta | FinResearch AI Project
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# =============================================================================
# STEP 0: ENVIRONMENT SETUP
# =============================================================================

from dotenv import load_dotenv

# Navigate to project root to find .env file
project_root = Path(__file__).resolve().parents[5]
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    load_dotenv()


def validate_environment() -> str:
    """Validates and returns the OpenAI API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 70)
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print("=" * 70)
        print("\nThe Assistants API requires a valid OpenAI API key.")
        print("Please add it to your .env file:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    return api_key


API_KEY = validate_environment()

# =============================================================================
# STEP 1: IMPORT OPENAI SDK
# =============================================================================
# CONCEPT: The OpenAI Python SDK (v1.0+) provides full support for the
# Assistants API through the client.beta.assistants namespace.
# The 'beta' prefix indicates this is a newer API still being refined.

from openai import OpenAI

# Initialize the client
client = OpenAI()

# For financial data
import yfinance as yf

# =============================================================================
# STEP 2: DEFINE OUR FINANCIAL RESEARCH TOOLS
# =============================================================================
# CONCEPT: Just like with Chat Completions, we define Python functions that
# the assistant can call. These functions fetch real market data.
#
# DIFFERENCE FROM CHAT COMPLETIONS:
# The tool execution loop is handled differently - we'll see this in the
# run processing logic below.


def get_stock_price(ticker: str) -> str:
    """Fetches current stock price for a ticker symbol."""
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        
        if history.empty:
            return json.dumps({"error": f"No data found for {ticker}"})
        
        price = history['Close'].iloc[-1]
        company_name = stock.info.get('shortName', ticker)
        
        return json.dumps({
            "ticker": ticker,
            "company": company_name,
            "price": round(price, 2),
            "currency": "USD"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_company_fundamentals(ticker: str) -> str:
    """Fetches comprehensive fundamental data for analysis."""
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract comprehensive data
        fundamentals = {
            "ticker": ticker,
            "company_name": info.get('shortName', 'N/A'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "market_cap": info.get('marketCap', 0),
            "enterprise_value": info.get('enterpriseValue', 0),
            "pe_ratio": info.get('trailingPE', None),
            "forward_pe": info.get('forwardPE', None),
            "peg_ratio": info.get('pegRatio', None),
            "price_to_book": info.get('priceToBook', None),
            "dividend_yield": info.get('dividendYield', None),
            "profit_margin": info.get('profitMargins', None),
            "operating_margin": info.get('operatingMargins', None),
            "roe": info.get('returnOnEquity', None),
            "roa": info.get('returnOnAssets', None),
            "revenue": info.get('totalRevenue', 0),
            "revenue_growth": info.get('revenueGrowth', None),
            "earnings_growth": info.get('earningsGrowth', None),
            "debt_to_equity": info.get('debtToEquity', None),
            "current_ratio": info.get('currentRatio', None),
            "free_cash_flow": info.get('freeCashflow', 0),
            "52_week_high": info.get('fiftyTwoWeekHigh', None),
            "52_week_low": info.get('fiftyTwoWeekLow', None),
            "50_day_average": info.get('fiftyDayAverage', None),
            "200_day_average": info.get('twoHundredDayAverage', None),
            "analyst_target": info.get('targetMeanPrice', None),
            "recommendation": info.get('recommendationKey', None),
            "business_summary": info.get('longBusinessSummary', 'N/A')[:500]
        }
        
        return json.dumps(fundamentals)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_stock_performance(ticker: str, period: str = "1y") -> str:
    """Analyzes stock performance over a given period."""
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        if history.empty:
            return json.dumps({"error": f"No data for {ticker}"})
        
        # Calculate performance metrics
        start_price = history['Close'].iloc[0]
        end_price = history['Close'].iloc[-1]
        high_price = history['High'].max()
        low_price = history['Low'].min()
        
        total_return = ((end_price - start_price) / start_price) * 100
        
        # Calculate volatility (annualized)
        daily_returns = history['Close'].pct_change().dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100
        
        # Calculate max drawdown
        rolling_max = history['Close'].expanding().max()
        drawdowns = (history['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        performance = {
            "ticker": ticker,
            "period": period,
            "start_price": round(start_price, 2),
            "end_price": round(end_price, 2),
            "total_return_pct": round(total_return, 2),
            "period_high": round(high_price, 2),
            "period_low": round(low_price, 2),
            "volatility_annualized_pct": round(volatility, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "trading_days": len(history)
        }
        
        return json.dumps(performance)
    except Exception as e:
        return json.dumps({"error": str(e)})


def compare_stocks(tickers: str) -> str:
    """Compares multiple stocks on key metrics."""
    try:
        # Parse comma-separated tickers
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        
        comparisons = []
        for ticker in ticker_list[:5]:  # Limit to 5 stocks
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="1y")
            
            if history.empty:
                continue
                
            # Calculate 1-year return
            start_price = history['Close'].iloc[0]
            end_price = history['Close'].iloc[-1]
            one_year_return = ((end_price - start_price) / start_price) * 100
            
            comparisons.append({
                "ticker": ticker,
                "company": info.get('shortName', ticker),
                "current_price": round(end_price, 2),
                "market_cap_b": round(info.get('marketCap', 0) / 1e9, 2),
                "pe_ratio": info.get('trailingPE', None),
                "1y_return_pct": round(one_year_return, 2),
                "dividend_yield_pct": round((info.get('dividendYield', 0) or 0) * 100, 2)
            })
        
        return json.dumps({"comparisons": comparisons})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# STEP 3: DEFINE TOOL SCHEMAS FOR THE ASSISTANT
# =============================================================================
# CONCEPT: The Assistants API uses the same JSON schema format as Chat
# Completions for function definitions. These schemas tell the assistant
# what tools are available and how to use them.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol. Use this for quick price checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'TSLA')"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_fundamentals",
            "description": "Get comprehensive fundamental data including financial ratios, margins, growth metrics, and company overview. Use this for in-depth company analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_performance",
            "description": "Analyze stock performance over a time period including returns, volatility, and drawdown. Use this for technical/performance analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period: 1mo, 3mo, 6mo, 1y, 2y, 5y",
                        "default": "1y"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare multiple stocks on key metrics. Use this when comparing 2+ stocks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "string",
                        "description": "Comma-separated ticker symbols (e.g., 'AAPL,MSFT,GOOGL')"
                    }
                },
                "required": ["tickers"]
            }
        }
    }
]

# Map function names to actual functions
FUNCTION_MAP = {
    "get_stock_price": get_stock_price,
    "get_company_fundamentals": get_company_fundamentals,
    "get_stock_performance": get_stock_performance,
    "compare_stocks": compare_stocks
}

# =============================================================================
# STEP 4: CREATE OR RETRIEVE THE ASSISTANT
# =============================================================================
# CONCEPT: Assistants are PERSISTENT - they live on OpenAI's servers.
# Once created, you can reuse the same assistant across multiple sessions.
# This is different from Chat Completions where you set up the system
# prompt every time.
#
# BEST PRACTICE: Store the assistant ID and reuse it rather than creating
# a new assistant every run (which would clutter your OpenAI dashboard).


def create_financial_assistant() -> str:
    """
    Creates a new Financial Research Assistant.
    
    Returns the assistant ID for future use.
    """
    print("🔧 Creating Financial Research Assistant...")
    
    assistant = client.beta.assistants.create(
        name="FinResearch AI - Financial Analyst",
        instructions="""You are an expert financial research analyst assistant. Your role is to:

1. ANALYZE stocks and companies using real market data
2. PROVIDE clear, actionable insights based on fundamental and technical analysis
3. COMPARE investments objectively using key metrics
4. EXPLAIN complex financial concepts in accessible terms

When analyzing stocks:
- Always use your tools to fetch real, current data
- Present data in a structured, easy-to-read format
- Highlight both opportunities and risks
- Include relevant ratios and their interpretations
- Compare metrics to industry averages when relevant

IMPORTANT DISCLAIMERS:
- This is for educational/informational purposes only
- Not financial advice - users should consult professionals
- Past performance doesn't guarantee future results
- Data may be delayed or incomplete

Format your responses with clear headers and bullet points for readability.""",
        
        model="gpt-4-turbo-preview",  # Use gpt-4 for better analysis
        tools=TOOL_DEFINITIONS
    )
    
    print(f"✅ Assistant created with ID: {assistant.id}")
    return assistant.id


def get_or_create_assistant() -> str:
    """
    Retrieves existing assistant or creates a new one.
    
    For this demo, we always create a new assistant to avoid ID management.
    In production, you'd store and reuse the assistant ID.
    """
    # In production, you might store the ID in a file or database:
    # assistant_id = load_from_storage()
    # if assistant_id:
    #     return assistant_id
    
    return create_financial_assistant()


# =============================================================================
# STEP 5: CREATE A CONVERSATION THREAD
# =============================================================================
# CONCEPT: A Thread is a conversation session. It:
# - Stores all messages in the conversation
# - Persists on OpenAI's servers
# - Can be retrieved later for continuation
#
# Think of it like a chat room that remembers everything.


def create_thread() -> str:
    """
    Creates a new conversation thread.
    
    Returns the thread ID.
    """
    thread = client.beta.threads.create()
    return thread.id


# =============================================================================
# STEP 6: SEND MESSAGES AND RUN THE ASSISTANT
# =============================================================================
# CONCEPT: The Run is where the magic happens. When you "run" an assistant
# on a thread, it:
# 1. Reads all messages in the thread
# 2. Decides what to do (possibly call tools)
# 3. Executes tools and processes results
# 4. Generates a final response
#
# The run goes through several statuses:
# - queued: Waiting to be processed
# - in_progress: Currently being processed
# - requires_action: Needs tool outputs from you
# - completed: Done!
# - failed: Something went wrong


def add_message_to_thread(thread_id: str, content: str) -> None:
    """Adds a user message to the thread."""
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )


def run_assistant(assistant_id: str, thread_id: str) -> str:
    """
    Runs the assistant on the thread and returns the response.
    
    This function handles the complete run lifecycle:
    1. Start the run
    2. Poll for completion (handling tool calls if needed)
    3. Extract and return the final response
    """
    # Start the run
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )
    
    print(f"🚀 Run started (ID: {run.id})")
    
    # Poll for completion
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        status = run.status
        print(f"   Status: {status}", end="\r")
        
        if status == "completed":
            print("\n✅ Run completed!")
            break
            
        elif status == "requires_action":
            # The assistant wants to call tools!
            print("\n🔧 Processing tool calls...")
            run = handle_tool_calls(thread_id, run)
            
        elif status in ["failed", "cancelled", "expired"]:
            print(f"\n❌ Run {status}")
            if run.last_error:
                print(f"   Error: {run.last_error.message}")
            return f"Error: Run {status}"
            
        else:
            # Still processing - wait a bit before polling again
            time.sleep(0.5)
    
    # Get the assistant's response
    messages = client.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc",
        limit=1
    )
    
    if messages.data:
        return messages.data[0].content[0].text.value
    return "No response generated"


def handle_tool_calls(thread_id: str, run) -> Any:
    """
    Handles tool calls requested by the assistant.
    
    When the assistant decides to use tools, we:
    1. Extract the requested tool calls
    2. Execute each tool with the provided arguments
    3. Submit the results back to the run
    """
    tool_outputs = []
    
    # Process each tool call
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        print(f"   📊 Calling: {function_name}({arguments})")
        
        # Execute the function
        if function_name in FUNCTION_MAP:
            result = FUNCTION_MAP[function_name](**arguments)
        else:
            result = json.dumps({"error": f"Unknown function: {function_name}"})
        
        tool_outputs.append({
            "tool_call_id": tool_call.id,
            "output": result
        })
    
    # Submit tool outputs back to the run
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run.id,
        tool_outputs=tool_outputs
    )
    
    return run


# =============================================================================
# STEP 7: HIGH-LEVEL CHAT FUNCTION
# =============================================================================

def chat(
    assistant_id: str,
    thread_id: str,
    user_message: str
) -> str:
    """
    Complete chat interaction: send message, run assistant, get response.
    """
    print(f"\n🧑 User: {user_message}")
    print("-" * 50)
    
    # Add user message to thread
    add_message_to_thread(thread_id, user_message)
    
    # Run the assistant and get response
    response = run_assistant(assistant_id, thread_id)
    
    print("-" * 50)
    print(f"🤖 Assistant:\n{response}")
    
    return response


# =============================================================================
# STEP 8: CLEANUP FUNCTION
# =============================================================================
# CONCEPT: Assistants persist on OpenAI's servers and count towards your
# usage. For demos, it's good practice to clean up when done.


def cleanup_assistant(assistant_id: str) -> None:
    """Deletes the assistant when we're done."""
    try:
        client.beta.assistants.delete(assistant_id)
        print(f"🧹 Cleaned up assistant {assistant_id}")
    except Exception as e:
        print(f"⚠️  Could not delete assistant: {e}")


# =============================================================================
# STEP 9: DEMO FUNCTIONS
# =============================================================================

def run_interactive_demo():
    """Interactive chat session with the Assistants API."""
    print("\n" + "=" * 70)
    print("🚀 OPENAI ASSISTANTS API DEMO: Financial Research Analyst")
    print("=" * 70)
    print("""
This demo shows OpenAI's Assistants API - a more sophisticated approach
to building AI agents with:

• Persistent Assistants - Defined once, used many times
• Managed Threads - Conversation history stored by OpenAI  
• Automatic Tool Handling - Multi-step reasoning built in

Available analysis capabilities:
  📈 Stock prices and quick lookups
  📊 Comprehensive fundamental analysis
  📉 Performance metrics and volatility
  ⚖️  Multi-stock comparisons

Example questions:
  • "Analyze Tesla's fundamentals and give me your assessment"
  • "Compare AAPL, MSFT, and GOOGL on key metrics"
  • "How has NVIDIA performed over the last year?"
  • "What's the current state of Amazon stock?"

Type 'quit' to exit.
""")
    print("=" * 70)
    
    # Create assistant and thread
    assistant_id = get_or_create_assistant()
    thread_id = create_thread()
    
    print(f"\n📝 Thread created: {thread_id}")
    print("Ready to chat!\n")
    
    try:
        while True:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            chat(assistant_id, thread_id, user_input)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        # Clean up
        print("\n👋 Goodbye!")
        cleanup_assistant(assistant_id)


def run_scripted_demo():
    """Scripted demonstration for presentations."""
    print("\n" + "=" * 70)
    print("🚀 OPENAI ASSISTANTS API: Scripted Demonstration")
    print("=" * 70)
    print("\nThis demonstrates the full capabilities of the Assistants API")
    print("for financial research and analysis.\n")
    
    # Create assistant and thread
    assistant_id = get_or_create_assistant()
    thread_id = create_thread()
    
    demo_queries = [
        "What's Apple's current stock price and give me a quick overview?",
        "I'm considering investing in NVIDIA. Give me a comprehensive fundamental analysis.",
        "Compare Tesla, Ford, and GM as investment options."
    ]
    
    try:
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*70}")
            print(f"📝 Demo Query {i}/{len(demo_queries)}")
            print("=" * 70)
            
            chat(assistant_id, thread_id, query)
            
            if i < len(demo_queries):
                print("\n⏳ Continuing to next query...")
                time.sleep(2)
        
        print("\n" + "=" * 70)
        print("🎓 DEMO COMPLETE: Key Takeaways")
        print("=" * 70)
        print("""
1. PERSISTENCE: The assistant remembers context across the conversation
   
2. AUTOMATIC TOOL HANDLING: Complex queries trigger multiple tool calls
   automatically - no manual loop management needed
   
3. RICH ANALYSIS: The assistant synthesizes data from multiple tools
   into coherent, structured responses
   
4. THREAD MANAGEMENT: OpenAI handles conversation history - you just
   add messages and run the assistant

5. PRODUCTION READY: Assistants can be reused across sessions,
   making them ideal for production applications

ASSISTANTS API ADVANTAGES:
  ✓ Cleaner code - less boilerplate
  ✓ Managed state - conversations persist
  ✓ Built-in tools - Code Interpreter, File Search
  ✓ Reliable tool handling - automatic retries
  ✓ Scalable - same assistant for all users
""")
        print("=" * 70)
        
    finally:
        cleanup_assistant(assistant_id)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OpenAI Assistants API Demo for Financial Research"
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run scripted demo instead of interactive mode"
    )
    args = parser.parse_args()
    
    if args.scripted:
        run_scripted_demo()
    else:
        run_interactive_demo()
