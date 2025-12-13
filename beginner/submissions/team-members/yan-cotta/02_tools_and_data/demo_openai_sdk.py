#!/usr/bin/env python3
"""
================================================================================
FinResearch AI - Beginner Track Demo: OpenAI SDK with Function Calling
================================================================================

PURPOSE:
    This script demonstrates how to use the OpenAI Python SDK directly to build
    an AI assistant that can call functions (tools) to fetch real financial data.
    
    This is the FOUNDATION of all agentic AI systems. Before using frameworks
    like CrewAI or LangChain, it's crucial to understand what's happening under
    the hood at the API level.

WHAT IS FUNCTION CALLING?
    Function calling (also called "tool use") is OpenAI's mechanism that allows
    the model to:
    1. Recognize when it needs external data to answer a question
    2. Request to call a specific function with specific arguments
    3. Receive the function's output and incorporate it into its response
    
    This turns a "dumb" text generator into a "smart" agent that can interact
    with the real world (APIs, databases, file systems, etc.).

KEY CONCEPTS COVERED:
    1. OpenAI Client Initialization - How to set up the API client
    2. Tool Definitions - How to describe functions so the model can use them
    3. Chat Completions API - The core API for conversational AI
    4. Tool Call Handling - Processing function requests from the model
    5. Multi-turn Conversations - Building context across multiple messages

ARCHITECTURE:
    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
    │  User Query  │ ───▶ │  OpenAI API  │ ───▶ │  Tool Call   │
    └──────────────┘      └──────────────┘      └──────────────┘
                                 │                      │
                                 ▼                      ▼
                          ┌──────────────┐      ┌──────────────┐
                          │   Response   │ ◀─── │ Yahoo Finance│
                          └──────────────┘      └──────────────┘

BEFORE RUNNING:
    1. Install dependencies: pip install -r requirements.txt
    2. Ensure your .env file contains: OPENAI_API_KEY=sk-...

Author: Yan Cotta | FinResearch AI Project
================================================================================
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional

# =============================================================================
# STEP 0: ENVIRONMENT SETUP & API KEY VALIDATION
# =============================================================================
# CONCEPT: Before making any API calls, we must:
# 1. Load environment variables from .env file
# 2. Validate that required API keys are present
# This prevents cryptic errors later in execution.

from dotenv import load_dotenv

# Navigate to project root to find .env file
# __file__ is the current script's path; we go up 4 directories to reach root
project_root = Path(__file__).resolve().parents[4]
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    load_dotenv()  # Try current directory as fallback
    print("⚠️  No .env file found at project root, trying current directory...")


def validate_api_key() -> str:
    """
    Validates the OpenAI API key is present and properly formatted.
    
    WHY THIS MATTERS:
    - OpenAI API calls fail silently or with confusing errors if key is missing
    - Early validation saves debugging time
    - Good practice for production code
    
    Returns:
        str: The validated API key
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 70)
        print("❌ ERROR: OPENAI_API_KEY not found in environment!")
        print("=" * 70)
        print("\nPlease ensure your .env file contains:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("\nOr set it directly:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    # Validate key format (OpenAI keys start with 'sk-')
    if not api_key.startswith("sk-"):
        print("⚠️  WARNING: API key doesn't start with 'sk-' - may be invalid")
    
    return api_key


# Validate before importing openai (which might auto-configure)
API_KEY = validate_api_key()

# =============================================================================
# STEP 1: IMPORT THE OPENAI SDK
# =============================================================================
# CONCEPT: The OpenAI Python SDK provides a clean interface to the API.
# We use the newer 'openai' package (v1.0+) which uses the OpenAI() client pattern.

from openai import OpenAI

# Initialize the OpenAI client
# The client will automatically use OPENAI_API_KEY from environment
client = OpenAI()

# =============================================================================
# STEP 2: IMPORT FINANCIAL DATA LIBRARY
# =============================================================================
# CONCEPT: yfinance is a popular library for fetching stock market data.
# It's free, requires no API key, and provides comprehensive market data.

import yfinance as yf

# =============================================================================
# STEP 3: DEFINE THE TOOLS (FUNCTIONS) THE AI CAN CALL
# =============================================================================
# CONCEPT: Tools are Python functions that the AI can request to call.
# We define them normally, then describe them to OpenAI in a specific format.
#
# IMPORTANT DESIGN PRINCIPLE:
# - Functions should be FOCUSED (do one thing well)
# - Functions should return STRINGS (easy for the model to parse)
# - Functions should handle ERRORS gracefully (don't crash the agent)


def get_stock_price(ticker: str) -> str:
    """
    Fetches the current stock price for a given ticker symbol.
    
    This function is called by the AI when it needs real-time price data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
        
    Returns:
        A formatted string with the stock price, or an error message
    """
    try:
        # Clean the input - users might type 'aapl' or ' AAPL '
        ticker = ticker.strip().upper()
        
        # Create a Ticker object for the stock
        stock = yf.Ticker(ticker)
        
        # Fetch the last day's trading data
        # 'period="1d"' gets data for the current/last trading day
        history = stock.history(period="1d")
        
        # Handle edge cases: invalid ticker or market closed with no data
        if history.empty:
            return f"❌ No data found for '{ticker}'. Is this a valid ticker symbol?"
        
        # Extract the closing price (last column, last row)
        current_price = history['Close'].iloc[-1]
        
        # Try to get the company name for a nicer response
        info = stock.info
        company_name = info.get('shortName', ticker)
        
        return f"📈 {company_name} ({ticker}): ${current_price:.2f} USD"
        
    except Exception as e:
        # Return error as string so the AI can tell the user what went wrong
        return f"❌ Error fetching price for '{ticker}': {str(e)}"


def get_company_info(ticker: str) -> str:
    """
    Fetches comprehensive company information including financials.
    
    This gives the AI context about what a company does, its size,
    and key financial metrics for analysis.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        A formatted string with company details
    """
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract key information with fallbacks for missing data
        # DEFENSIVE CODING: Not all stocks have all fields
        company_name = info.get('shortName', 'Unknown Company')
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 'N/A')
        week_52_high = info.get('fiftyTwoWeekHigh', 'N/A')
        week_52_low = info.get('fiftyTwoWeekLow', 'N/A')
        description = info.get('longBusinessSummary', 'No description available.')
        
        # Format market cap for readability (1,234,567,890 → $1.23B)
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_str = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_str = f"${market_cap:,.0f}"
        else:
            market_cap_str = "N/A"
        
        # Build a structured response
        return f"""
🏢 **{company_name} ({ticker})**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Sector: {sector}
🏭 Industry: {industry}
💰 Market Cap: {market_cap_str}
📈 P/E Ratio: {pe_ratio}
📊 52-Week Range: ${week_52_low} - ${week_52_high}

📝 **About:**
{description[:500]}{'...' if len(description) > 500 else ''}
"""
    
    except Exception as e:
        return f"❌ Error fetching info for '{ticker}': {str(e)}"


# =============================================================================
# STEP 4: DEFINE TOOL SCHEMAS FOR OPENAI
# =============================================================================
# CONCEPT: OpenAI needs to know what tools are available and how to use them.
# We describe each tool using a JSON Schema format:
# - name: What to call this tool
# - description: When should the AI use this? (CRITICAL for good tool selection)
# - parameters: What inputs does the tool need?
#
# PRO TIP: The 'description' field is the most important! The AI reads this to
# decide when to use the tool. Be specific about what the tool does and when
# it's appropriate to use it.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Get the current stock price for a given ticker symbol. "
                "Use this when the user asks about a stock's current price, "
                "value, or how much a stock costs. Examples: 'What's Apple's "
                "stock price?', 'How much is TSLA trading at?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": (
                            "The stock ticker symbol (e.g., 'AAPL' for Apple, "
                            "'TSLA' for Tesla, 'GOOGL' for Google)"
                        )
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": (
                "Get detailed information about a company including its sector, "
                "industry, market cap, P/E ratio, 52-week range, and business "
                "description. Use this when the user asks about a company's "
                "background, what they do, or wants a company overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]


# =============================================================================
# STEP 5: CREATE THE TOOL EXECUTION DISPATCHER
# =============================================================================
# CONCEPT: When the AI requests to call a tool, we need to:
# 1. Parse the tool name from the request
# 2. Parse the arguments from the request
# 3. Call the actual Python function
# 4. Return the result to the AI
#
# This is a "dispatcher" pattern - it routes requests to the right function.

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Executes a tool by name with the given arguments.
    
    This function acts as a dispatcher that connects OpenAI's tool requests
    to our actual Python functions.
    
    Args:
        tool_name: The name of the tool to call
        tool_args: Dictionary of arguments for the tool
        
    Returns:
        The result from the tool as a string
    """
    # Map tool names to actual functions
    tool_map = {
        "get_stock_price": get_stock_price,
        "get_company_info": get_company_info,
    }
    
    # Look up the function
    if tool_name not in tool_map:
        return f"❌ Unknown tool: {tool_name}"
    
    # Call the function with the provided arguments
    # **tool_args unpacks the dictionary into keyword arguments
    return tool_map[tool_name](**tool_args)


# =============================================================================
# STEP 6: CREATE THE MAIN CHAT FUNCTION WITH TOOL HANDLING
# =============================================================================
# CONCEPT: This is the core of our agent. The chat function:
# 1. Sends the user's message to OpenAI
# 2. Checks if the AI wants to call any tools
# 3. If yes, executes the tools and sends results back
# 4. Gets the final response with tool results incorporated
#
# This is called "agentic behavior" - the AI decides what actions to take!

def chat_with_tools(user_message: str, conversation_history: list = None) -> str:
    """
    Sends a message to OpenAI and handles any tool calls the model requests.
    
    This function implements the full "agentic loop":
    1. User message → OpenAI
    2. OpenAI → Tool requests (optional)
    3. Tool execution → Results back to OpenAI
    4. OpenAI → Final response
    
    Args:
        user_message: The user's question or request
        conversation_history: Optional list of previous messages for context
        
    Returns:
        The assistant's final response as a string
    """
    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = []
    
    # Add system message if this is a new conversation
    # SYSTEM PROMPT: Sets the AI's personality and behavior rules
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": (
                "You are a helpful financial research assistant. You have access "
                "to real-time stock market data through your tools. When users ask "
                "about stocks, use your tools to get accurate, up-to-date information. "
                "Always be clear about what data you're providing and remind users "
                "that stock prices change frequently. Be concise but informative."
            )
        })
    
    # Add the user's message to the conversation
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    print(f"\n🧑 User: {user_message}")
    print("🤖 Thinking...", end="", flush=True)
    
    # ----- STEP A: Initial API Call -----
    # We send the conversation to OpenAI with our tool definitions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Using 3.5 for cost efficiency; use gpt-4 for better reasoning
        messages=conversation_history,
        tools=TOOLS,  # Our function definitions
        tool_choice="auto"  # Let the model decide whether to use tools
    )
    
    # Get the assistant's response
    assistant_message = response.choices[0].message
    
    # ----- STEP B: Check for Tool Calls -----
    # The model might want to call one or more tools before responding
    if assistant_message.tool_calls:
        print(f" (calling {len(assistant_message.tool_calls)} tool(s))")
        
        # Add the assistant's message (with tool calls) to history
        conversation_history.append(assistant_message)
        
        # Execute each tool and collect results
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"   🔧 Calling: {tool_name}({tool_args})")
            
            # Execute the tool
            tool_result = execute_tool(tool_name, tool_args)
            
            print(f"   📊 Result: {tool_result[:100]}...")
            
            # Add the tool result to conversation history
            # IMPORTANT: We must include the tool_call_id so OpenAI knows
            # which tool call this result corresponds to
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
        
        # ----- STEP C: Get Final Response -----
        # Now we send the conversation (including tool results) back to OpenAI
        # The model will incorporate the tool results into its final response
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history
        )
        
        final_message = final_response.choices[0].message
        conversation_history.append(final_message)
        
        print(f"\n🤖 Assistant: {final_message.content}")
        return final_message.content
    
    else:
        # No tool calls - just a regular response
        print(" (no tools needed)")
        conversation_history.append(assistant_message)
        
        print(f"\n🤖 Assistant: {assistant_message.content}")
        return assistant_message.content


# =============================================================================
# STEP 7: INTERACTIVE DEMO
# =============================================================================
# CONCEPT: This creates a simple interactive loop where you can chat with
# the AI assistant and see tool calling in action.

def run_interactive_demo():
    """
    Runs an interactive command-line chat session.
    
    This demonstrates:
    - Multi-turn conversations (context is preserved)
    - Tool calling in action
    - Error handling
    """
    print("\n" + "=" * 70)
    print("🚀 OPENAI SDK DEMO: Financial Research Assistant with Tool Calling")
    print("=" * 70)
    print("\nThis demo shows how OpenAI's function calling works.")
    print("Try asking about stock prices or company information!")
    print("\nExample questions:")
    print("  • What's Apple's current stock price?")
    print("  • Tell me about Tesla as a company")
    print("  • Compare the prices of AAPL and MSFT")
    print("\nType 'quit' to exit.\n")
    print("=" * 70)
    
    conversation = []
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for trying the demo.\n")
                break
            
            chat_with_tools(user_input, conversation)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Try again or type 'quit' to exit.")


# =============================================================================
# STEP 8: SCRIPTED DEMO (For Presentations)
# =============================================================================
# CONCEPT: For team presentations, a scripted demo is more reliable than
# live typing. This runs through predetermined questions.

def run_scripted_demo():
    """
    Runs a scripted demo with predetermined questions.
    
    Perfect for presentations where you want predictable behavior.
    """
    print("\n" + "=" * 70)
    print("🚀 OPENAI SDK DEMO: Scripted Demonstration")
    print("=" * 70)
    print("\nRunning through example queries to demonstrate tool calling...\n")
    
    # Scripted questions to demonstrate different capabilities
    demo_questions = [
        "What's Apple's current stock price?",
        "Tell me about Tesla - what does the company do and what's their market cap?",
        "What's the current price of Microsoft stock?"
    ]
    
    conversation = []
    
    for question in demo_questions:
        print("\n" + "-" * 50)
        chat_with_tools(question, conversation)
        print("-" * 50)
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. The AI automatically decided when to use tools")
    print("  2. Tool results were incorporated into natural language responses")
    print("  3. Conversation context was maintained across questions")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAI SDK Financial Assistant Demo")
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
