#!/usr/bin/env python3
"""
================================================================================
FINRESEARCH AI - WEEK 2: DATA FETCHER MODULE
================================================================================

LEARNING OBJECTIVES:
    1. Build reusable functions that return clean, structured data (not prints!)
    2. Implement proper error handling for unreliable external APIs
    3. Learn to combine multiple data sources into a unified response
    4. Understand the importance of data normalization for downstream processing

WHAT WE'RE BUILDING:
    This module provides two core data-fetching capabilities:
    
    1. MARKET DATA FETCHER (yfinance)
       - Current price, market cap, P/E ratio
       - 52-week high/low, volume data
       - Company sector and industry
    
    2. NEWS FETCHER (DuckDuckGo Search)
       - Recent news articles about a company
       - Headline, URL, and snippet for each result
       - Free to use - no API key required!

TEACHING NOTE - WHY RETURN DICTIONARIES, NOT PRINT STATEMENTS?
    In production code, functions should RETURN data, not print it. Why?
    
    BAD:  print(f"Price: {price}")  -> Data is lost! Can't be reused.
    GOOD: return {"price": price}  -> Data can be stored, processed, displayed.
    
    This separation of concerns means:
    - The FETCHER is responsible for getting data
    - The CALLER decides what to do with it (print, save, send to AI, etc.)

DEPENDENCIES:
    pip install yfinance duckduckgo-search python-dotenv

Author: Yan Cotta | FinResearch AI Project
Week: 2 - Data Retrieval & Error Handling
================================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# TEACHING NOTE: Even though these tools don't require API keys, we maintain
# consistent environment setup practices. This makes it easy to add paid APIs
# later without restructuring the code.

from dotenv import load_dotenv

# Find and load .env file from project root
project_root = Path(__file__).resolve().parents[4]
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Environment loaded from: {env_path}")
else:
    load_dotenv()  # Try current directory


# =============================================================================
# STEP 1: IMPORT DATA LIBRARIES
# =============================================================================
# TEACHING NOTE: We import inside a try/except block to give friendly error
# messages if dependencies are missing. This is especially helpful for juniors
# who might forget to install requirements.

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed!")
    print("   Run: pip install yfinance")
    sys.exit(1)

try:
    from duckduckgo_search import DDGS
except ImportError:
    print("ERROR: duckduckgo-search not installed!")
    print("   Run: pip install duckduckgo-search")
    sys.exit(1)


# =============================================================================
# STEP 2: DEFINE RESULT TYPES (Type Hints for Clarity)
# =============================================================================
# TEACHING NOTE: Type hints make code self-documenting. When you see:
#     def get_data() -> Dict[str, Any]:
# You immediately know the function returns a dictionary.
#
# We define type aliases here for complex types to improve readability.

MarketDataResult = Dict[str, Any]
NewsResult = Dict[str, Any]
CombinedResult = Dict[str, Any]


# =============================================================================
# STEP 3: MARKET DATA FETCHER (yfinance)
# =============================================================================
# TEACHING NOTE - FUNCTION DESIGN PRINCIPLES:
#   1. Single Responsibility: This function ONLY fetches market data
#   2. Defensive Programming: Assume everything can fail
#   3. Consistent Return Format: Always return same structure, even on error
#   4. No Side Effects: Don't print, don't modify global state

def fetch_market_data(ticker: str) -> MarketDataResult:
    """
    Fetches comprehensive market data for a given stock ticker.
    
    This function demonstrates several important concepts:
    - Input validation (cleaning user input)
    - Error handling with informative messages
    - Consistent return format regardless of success/failure
    - Data normalization (converting raw API data to clean format)
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')
                Can be lowercase or have extra spaces - we'll clean it.
    
    Returns:
        A dictionary with the following structure:
        {
            "success": bool,           # Did the fetch succeed?
            "ticker": str,             # The normalized ticker symbol
            "timestamp": str,          # When this data was fetched
            "data": {...} or None,     # The actual market data (if success)
            "error": str or None       # Error message (if failure)
        }
    
    Example:
        >>> result = fetch_market_data("AAPL")
        >>> if result["success"]:
        ...     print(f"Price: ${result['data']['current_price']}")
        ... else:
        ...     print(f"Error: {result['error']}")
    """
    
    # -------------------------------------------------------------------------
    # STEP 3.1: INPUT VALIDATION & NORMALIZATION
    # -------------------------------------------------------------------------
    # TEACHING NOTE: Never trust user input! Users might type:
    #   - "aapl" (lowercase)
    #   - " AAPL " (extra spaces)
    #   - "AAPL\n" (newlines from copy-paste)
    # We normalize everything to a consistent format.
    
    # Clean the ticker: remove whitespace, convert to uppercase
    ticker = ticker.strip().upper()
    
    # Basic validation: ticker should be 1-5 alphanumeric characters
    if not ticker or not ticker.isalnum() or len(ticker) > 5:
        return {
            "success": False,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "data": None,
            "error": f"Invalid ticker format: '{ticker}'. Expected 1-5 alphanumeric characters."
        }
    
    # -------------------------------------------------------------------------
    # STEP 3.2: FETCH DATA FROM YFINANCE
    # -------------------------------------------------------------------------
    # TEACHING NOTE: We wrap the entire fetch in a try/except because:
    #   - Network requests can fail (no internet, timeout, etc.)
    #   - The ticker might not exist
    #   - Yahoo Finance might rate-limit us
    #   - The response format might change unexpectedly
    
    try:
        # Create a Ticker object - this doesn't make any API calls yet
        stock = yf.Ticker(ticker)
        
        # Fetch company info - this DOES make an API call
        info = stock.info
        
        # TEACHING NOTE: yfinance returns an empty dict or minimal data for
        # invalid tickers. We check for a key field to validate.
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None:
            # Try to get historical data as a fallback validation
            history = stock.history(period="1d")
            if history.empty:
                return {
                    "success": False,
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "data": None,
                    "error": f"Ticker '{ticker}' not found. Please verify the symbol."
                }
        
        # -------------------------------------------------------------------------
        # STEP 3.3: EXTRACT AND NORMALIZE DATA
        # -------------------------------------------------------------------------
        # TEACHING NOTE: Raw API responses are often messy. We:
        #   1. Extract only the fields we need
        #   2. Provide sensible defaults for missing data
        #   3. Format values consistently (e.g., round prices to 2 decimals)
        
        # Get current price (different fields depending on market status)
        current_price = (
            info.get('regularMarketPrice') or 
            info.get('currentPrice') or 
            info.get('previousClose')
        )
        
        # Get historical data for additional context
        history = stock.history(period="5d")
        
        # Calculate price change if we have history
        if not history.empty and len(history) >= 2:
            prev_close = history['Close'].iloc[-2]
            today_close = history['Close'].iloc[-1]
            price_change = today_close - prev_close
            price_change_pct = (price_change / prev_close) * 100
        else:
            price_change = None
            price_change_pct = None
        
        # Format market cap for readability
        market_cap = info.get('marketCap', 0)
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_formatted = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_formatted = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_formatted = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_formatted = f"${market_cap:,.0f}"
        else:
            market_cap_formatted = "N/A"
        
        # Build the normalized data structure
        data = {
            # Company Information
            "company_name": info.get('shortName') or info.get('longName') or ticker,
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "website": info.get('website', 'N/A'),
            "description": info.get('longBusinessSummary', 'No description available.'),
            
            # Price Data
            "current_price": round(current_price, 2) if current_price else None,
            "currency": info.get('currency', 'USD'),
            "price_change": round(price_change, 2) if price_change else None,
            "price_change_percent": round(price_change_pct, 2) if price_change_pct else None,
            
            # Market Data
            "market_cap": market_cap,
            "market_cap_formatted": market_cap_formatted,
            "volume": info.get('volume', 'N/A'),
            "average_volume": info.get('averageVolume', 'N/A'),
            
            # Valuation Metrics
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "forward_pe": info.get('forwardPE', 'N/A'),
            "eps": info.get('trailingEps', 'N/A'),
            "beta": info.get('beta', 'N/A'),
            
            # Price Range
            "day_high": info.get('dayHigh', 'N/A'),
            "day_low": info.get('dayLow', 'N/A'),
            "fifty_two_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "fifty_two_week_low": info.get('fiftyTwoWeekLow', 'N/A'),
            
            # Dividends
            "dividend_yield": info.get('dividendYield', 'N/A'),
            "dividend_rate": info.get('dividendRate', 'N/A'),
        }
        
        return {
            "success": True,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "error": None
        }
        
    except Exception as e:
        # -------------------------------------------------------------------------
        # STEP 3.4: ERROR HANDLING
        # -------------------------------------------------------------------------
        # TEACHING NOTE: We catch ALL exceptions here because:
        #   1. We don't want the entire program to crash
        #   2. We want to return a consistent format
        #   3. We log the error type for debugging
        #
        # In production, you might want to:
        #   - Log errors to a file or monitoring service
        #   - Retry transient failures
        #   - Return cached data if available
        
        return {
            "success": False,
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "data": None,
            "error": f"{type(e).__name__}: {str(e)}"
        }


# =============================================================================
# STEP 4: NEWS FETCHER (DuckDuckGo Search)
# =============================================================================
# TEACHING NOTE - WHY DUCKDUCKGO?
#   - FREE: No API key required (great for learning!)
#   - SIMPLE: Easy to use Python library
#   - SUFFICIENT: Good enough for basic news discovery
#
#   For production, you might upgrade to:
#   - Tavily (designed for AI agents)
#   - NewsAPI (dedicated news aggregator)
#   - Google Custom Search (more control)

def fetch_company_news(
    query: str, 
    max_results: int = 5,
    region: str = "wt-wt"
) -> NewsResult:
    """
    Fetches recent news articles about a company using DuckDuckGo search.
    
    This function demonstrates:
    - Query construction for better search results
    - Result limiting to avoid overwhelming the AI
    - Graceful handling of rate limits and timeouts
    
    Args:
        query: The search query (e.g., "Apple Inc" or "AAPL stock news")
        max_results: Maximum number of articles to return (default: 5)
                    TEACHING NOTE: More isn't always better! AI models have
                    context limits, and too many articles = confusion.
        region: DuckDuckGo region code (default: "wt-wt" for no region bias)
    
    Returns:
        A dictionary with the following structure:
        {
            "success": bool,
            "query": str,
            "timestamp": str,
            "articles": [...] or [],
            "article_count": int,
            "error": str or None
        }
    
    Example:
        >>> result = fetch_company_news("Tesla stock news")
        >>> for article in result["articles"]:
        ...     print(f"• {article['title']}")
    """
    
    # -------------------------------------------------------------------------
    # STEP 4.1: INPUT VALIDATION
    # -------------------------------------------------------------------------
    
    if not query or not query.strip():
        return {
            "success": False,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "articles": [],
            "article_count": 0,
            "error": "Empty search query provided."
        }
    
    # Clean the query
    query = query.strip()
    
    # -------------------------------------------------------------------------
    # STEP 4.2: ENHANCE QUERY FOR BETTER RESULTS
    # -------------------------------------------------------------------------
    # TEACHING NOTE: Raw company names often return irrelevant results.
    # Adding context like "stock news" or "financial" improves relevance.
    
    # Check if query already has financial context
    financial_keywords = ['stock', 'share', 'market', 'financial', 'earnings', 'investor']
    has_financial_context = any(kw in query.lower() for kw in financial_keywords)
    
    if not has_financial_context:
        # Add financial context to improve relevance
        enhanced_query = f"{query} stock news"
    else:
        enhanced_query = query
    
    # -------------------------------------------------------------------------
    # STEP 4.3: PERFORM THE SEARCH
    # -------------------------------------------------------------------------
    
    try:
        # TEACHING NOTE: DDGS() creates a new search session.
        # We use it as a context manager to ensure proper cleanup.
        
        articles = []
        
        with DDGS() as ddgs:
            # Search for news articles
            # TEACHING NOTE: We request more than max_results because some
            # results might be low quality and we'll filter them.
            
            results = list(ddgs.news(
                keywords=enhanced_query,
                region=region,
                max_results=max_results + 3  # Request a few extra for filtering
            ))
        
        # -------------------------------------------------------------------------
        # STEP 4.4: PROCESS AND NORMALIZE RESULTS
        # -------------------------------------------------------------------------
        # TEACHING NOTE: Raw search results are messy. We extract only what we
        # need and ensure consistent formatting.
        
        for result in results[:max_results]:  # Limit to requested amount
            article = {
                "title": result.get('title', 'No title'),
                "url": result.get('url', ''),
                "snippet": result.get('body', 'No description available.'),
                "source": result.get('source', 'Unknown'),
                "published": result.get('date', 'Unknown date'),
            }
            
            # Basic quality filter: skip articles without URLs
            if article['url']:
                articles.append(article)
        
        return {
            "success": True,
            "query": query,
            "enhanced_query": enhanced_query,
            "timestamp": datetime.now().isoformat(),
            "articles": articles,
            "article_count": len(articles),
            "error": None
        }
        
    except Exception as e:
        # TEACHING NOTE: DuckDuckGo can rate-limit or block requests.
        # We handle this gracefully instead of crashing.
        
        return {
            "success": False,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "articles": [],
            "article_count": 0,
            "error": f"{type(e).__name__}: {str(e)}"
        }


# =============================================================================
# STEP 5: COMBINED DATA FETCHER
# =============================================================================
# TEACHING NOTE - COMPOSITION PATTERN:
#   This function COMPOSES the two individual fetchers into a single call.
#   This is a common pattern in software engineering:
#   - Individual functions do ONE thing well
#   - Higher-level functions COMBINE them for convenience
#   
#   The caller can choose to use:
#   - fetch_market_data() alone (just need prices)
#   - fetch_company_news() alone (just need news)
#   - fetch_all_data() (need everything)

def fetch_all_data(ticker: str, news_count: int = 5) -> CombinedResult:
    """
    Fetches both market data and news for a given stock ticker.
    
    This is a convenience function that combines market data and news
    fetching into a single call. It's designed for use cases where you
    need a complete picture of a company.
    
    TEACHING NOTE - ERROR INDEPENDENCE:
        If one data source fails, we still return data from the other.
        This is called "partial failure handling" - don't let one bad
        apple spoil the bunch!
    
    Args:
        ticker: Stock ticker symbol
        news_count: Number of news articles to fetch (default: 5)
    
    Returns:
        A dictionary combining market data and news:
        {
            "ticker": str,
            "timestamp": str,
            "market_data": {...},  # Result from fetch_market_data
            "news": {...},         # Result from fetch_company_news
            "overall_success": bool  # True if at least one source succeeded
        }
    """
    
    # Normalize ticker for consistent use
    ticker = ticker.strip().upper()
    
    # Fetch market data
    market_result = fetch_market_data(ticker)
    
    # Construct news query using company name if available
    if market_result["success"] and market_result["data"]:
        company_name = market_result["data"].get("company_name", ticker)
        news_query = f"{company_name} ({ticker})"
    else:
        news_query = f"{ticker} stock"
    
    # Fetch news
    news_result = fetch_company_news(news_query, max_results=news_count)
    
    # Determine overall success (at least one source worked)
    overall_success = market_result["success"] or news_result["success"]
    
    return {
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "market_data": market_result,
        "news": news_result,
        "overall_success": overall_success
    }


# =============================================================================
# STEP 6: PRETTY PRINTING (FOR DEMOS AND DEBUGGING)
# =============================================================================
# TEACHING NOTE: These functions are for DISPLAY purposes only.
# Notice how they're separate from the data-fetching functions.
# This is the "Separation of Concerns" principle in action!

def format_market_data(result: MarketDataResult) -> str:
    """
    Formats market data result into a human-readable string.
    
    TEACHING NOTE: This is a pure formatting function - no data fetching.
    It takes data IN and returns a string OUT. Easy to test!
    """
    if not result["success"]:
        return f"ERROR: Error fetching data for {result['ticker']}: {result['error']}"
    
    data = result["data"]
    
    # Build price change indicator
    if data["price_change"] is not None:
        if data["price_change"] >= 0:
            change_str = f"UP +${data['price_change']:.2f} (+{data['price_change_percent']:.2f}%)"
        else:
            change_str = f"DOWN ${data['price_change']:.2f} ({data['price_change_percent']:.2f}%)"
    else:
        change_str = "N/A"
    
    return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  MARKET DATA: {data['company_name'][:45]:<45} ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ticker: {result['ticker']:<12}  │  Sector: {str(data['sector'])[:25]:<25}  ║
║  Price:  ${data['current_price']:<10}  │  Industry: {str(data['industry'])[:23]:<23}  ║
║  Change: {change_str:<49} ║
╠══════════════════════════════════════════════════════════════════════╣
║  VALUATION METRICS                                                   ║
║  ─────────────────                                                   ║
║  Market Cap: {data['market_cap_formatted']:<15}  P/E Ratio: {str(data['pe_ratio'])[:10]:<10}         ║
║  EPS: {str(data['eps']):<15}           Beta: {str(data['beta'])[:10]:<10}              ║
╠══════════════════════════════════════════════════════════════════════╣
║  PRICE RANGE                                                         ║
║  ───────────                                                         ║
║  Day:     ${str(data['day_low'])[:8]:<8} - ${str(data['day_high'])[:8]:<8}                          ║
║  52-Week: ${str(data['fifty_two_week_low'])[:8]:<8} - ${str(data['fifty_two_week_high'])[:8]:<8}                          ║
╚══════════════════════════════════════════════════════════════════════╝
    """


def format_news(result: NewsResult) -> str:
    """Formats news result into a human-readable string."""
    if not result["success"]:
        return f"ERROR: Error fetching news: {result['error']}"
    
    if not result["articles"]:
        return f"No news found for: {result['query']}"
    
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        f"║  RECENT NEWS: {result['query'][:50]:<50} ║",
        "╠══════════════════════════════════════════════════════════════════════╣"
    ]
    
    for i, article in enumerate(result["articles"], 1):
        # Truncate long titles
        title = article['title'][:60] + "..." if len(article['title']) > 60 else article['title']
        source = article['source'][:15]
        
        lines.append(f"║  {i}. {title:<64} ║")
        lines.append(f"║     Source: {source:<20}  Date: {article['published'][:20]:<20} ║")
        if i < len(result["articles"]):
            lines.append("║  ──────────────────────────────────────────────────────────────────  ║")
    
    lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    
    return "\n".join(lines)


# =============================================================================
# STEP 7: DEMO / MAIN ENTRY POINT
# =============================================================================
# TEACHING NOTE: The demo section shows how to USE the functions we built.
# This serves as both documentation and a manual test.

def run_demo():
    """
    Demonstrates the data fetching capabilities.
    
    This demo shows:
    1. Fetching market data for a valid ticker
    2. Fetching news for a company
    3. Handling invalid tickers gracefully
    4. Using the combined fetcher
    """
    
    print("\n" + "=" * 72)
    print("  FINRESEARCH AI - WEEK 2: DATA FETCHER DEMO")
    print("=" * 72)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    
    # -------------------------------------------------------------------------
    # DEMO 1: Fetch Market Data
    # -------------------------------------------------------------------------
    print("\n\nDEMO 1: Fetching Market Data for AAPL")
    print("-" * 72)
    
    market_result = fetch_market_data("AAPL")
    print(format_market_data(market_result))
    
    # -------------------------------------------------------------------------
    # DEMO 2: Fetch News
    # -------------------------------------------------------------------------
    print("\n\nDEMO 2: Fetching News for Apple")
    print("-" * 72)
    
    news_result = fetch_company_news("Apple Inc AAPL", max_results=3)
    print(format_news(news_result))
    
    # -------------------------------------------------------------------------
    # DEMO 3: Error Handling - Invalid Ticker
    # -------------------------------------------------------------------------
    print("\n\nDEMO 3: Testing Error Handling (Invalid Ticker)")
    print("-" * 72)
    
    invalid_result = fetch_market_data("XYZNOTREAL123")
    print(format_market_data(invalid_result))
    
    # -------------------------------------------------------------------------
    # DEMO 4: Combined Fetcher
    # -------------------------------------------------------------------------
    print("\n\nDEMO 4: Combined Data Fetch for MSFT")
    print("-" * 72)
    
    combined_result = fetch_all_data("MSFT", news_count=3)
    print(format_market_data(combined_result["market_data"]))
    print(format_news(combined_result["news"]))
    
    # -------------------------------------------------------------------------
    # DEMO 5: Show Raw JSON Structure
    # -------------------------------------------------------------------------
    print("\n\nDEMO 5: Raw JSON Structure (for AI consumption)")
    print("-" * 72)
    print("TEACHING NOTE: This is what your AI agent will receive!")
    print("-" * 72)
    
    import json
    
    # Show a simplified version of the raw data
    sample = fetch_market_data("GOOGL")
    if sample["success"]:
        simplified = {
            "success": sample["success"],
            "ticker": sample["ticker"],
            "company_name": sample["data"]["company_name"],
            "current_price": sample["data"]["current_price"],
            "market_cap_formatted": sample["data"]["market_cap_formatted"],
            "pe_ratio": sample["data"]["pe_ratio"],
            "sector": sample["data"]["sector"]
        }
        print(json.dumps(simplified, indent=2))
    
    print("\n" + "=" * 72)
    print("  DEMO COMPLETE")
    print("=" * 72)
    print("\n  Key Takeaways:")
    print("  1. Functions return structured dictionaries, not print output")
    print("  2. All results have a 'success' field for easy error checking")
    print("  3. Error messages are informative and actionable")
    print("  4. Data is normalized for consistent downstream processing")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_demo()
