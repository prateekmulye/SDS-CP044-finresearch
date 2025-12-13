"""
Financial Data Tool - Wrapper for Yahoo Finance API.

Provides robust access to stock prices, valuation metrics, and fundamental data
with comprehensive error handling and structured output.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from crewai.tools import BaseTool
from pydantic import Field

try:
    import yfinance as yf
except ImportError:
    yf = None  # Handle gracefully in tool execution

from src.tools.base import ToolError


logger = logging.getLogger(__name__)


class FinancialDataTool(BaseTool):
    """
    Tool for fetching quantitative financial data from Yahoo Finance.
    
    Provides access to:
    - Current and historical stock prices
    - Valuation metrics (P/E, P/B, EV/EBITDA)
    - Fundamental data (market cap, revenue, margins)
    - Trading information (volume, beta, 52-week range)
    """
    
    name: str = "financial_data_tool"
    description: str = (
        "Fetches quantitative financial data for a stock ticker symbol. "
        "Returns current price, valuation metrics (P/E, P/B, EPS), "
        "market cap, 52-week range, volume, and other financial metrics. "
        "Input: stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL'). "
        "Use this for NUMBERS and METRICS, not news or sentiment."
    )
    
    def _run(self, ticker: str) -> str:
        """
        Fetch financial data for the specified ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Formatted string with financial data or error message
        """
        if yf is None:
            logger.error("yfinance library not installed")
            return "ERROR: yfinance library is not installed. Run: pip install yfinance"
        
        ticker = self._normalize_ticker(ticker)
        
        try:
            data = self._fetch_data(ticker)
            return self._format_output(ticker, data)
        except ToolError as e:
            logger.warning(f"Tool error for {ticker}: {e.message}")
            return f"ERROR: {e.message}"
        except Exception as e:
            logger.exception(f"Unexpected error fetching data for {ticker}")
            return f"ERROR: Unexpected error fetching data for '{ticker}': {type(e).__name__}"
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol input."""
        if not ticker:
            raise ToolError(self.name, "Empty ticker symbol provided")
        return ticker.strip().upper()
    
    def _fetch_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch raw data from Yahoo Finance.
        
        Args:
            ticker: Normalized ticker symbol
            
        Returns:
            Dictionary with financial data
            
        Raises:
            ToolError: If data cannot be retrieved
        """
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Validate we got meaningful data
        if not info or info.get('regularMarketPrice') is None:
            history = stock.history(period="5d")
            if history.empty:
                raise ToolError(
                    self.name,
                    f"No data found for ticker '{ticker}'. Verify the symbol is correct."
                )
            # Use historical data as fallback for price
            info['regularMarketPrice'] = history['Close'].iloc[-1]
        
        return self._extract_metrics(info)
    
    def _extract_metrics(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize relevant metrics from raw data."""
        
        def safe_get(key: str, default: Any = None) -> Any:
            """Safely get value with default."""
            value = info.get(key)
            return value if value is not None else default
        
        def format_large_number(value: Optional[float]) -> str:
            """Format large numbers for readability."""
            if value is None or value == 0:
                return "N/A"
            if value >= 1e12:
                return f"${value/1e12:.2f}T"
            if value >= 1e9:
                return f"${value/1e9:.2f}B"
            if value >= 1e6:
                return f"${value/1e6:.2f}M"
            return f"${value:,.0f}"
        
        return {
            # Company Info
            "company_name": safe_get('shortName') or safe_get('longName', 'Unknown'),
            "sector": safe_get('sector', 'N/A'),
            "industry": safe_get('industry', 'N/A'),
            
            # Price Data
            "current_price": safe_get('regularMarketPrice') or safe_get('currentPrice'),
            "previous_close": safe_get('previousClose'),
            "currency": safe_get('currency', 'USD'),
            "day_high": safe_get('dayHigh'),
            "day_low": safe_get('dayLow'),
            "fifty_two_week_high": safe_get('fiftyTwoWeekHigh'),
            "fifty_two_week_low": safe_get('fiftyTwoWeekLow'),
            
            # Valuation
            "market_cap": safe_get('marketCap'),
            "market_cap_formatted": format_large_number(safe_get('marketCap')),
            "trailing_pe": safe_get('trailingPE'),
            "forward_pe": safe_get('forwardPE'),
            "price_to_book": safe_get('priceToBook'),
            "peg_ratio": safe_get('pegRatio'),
            "enterprise_value": safe_get('enterpriseValue'),
            "ev_to_ebitda": safe_get('enterpriseToEbitda'),
            
            # Fundamentals
            "eps_ttm": safe_get('trailingEps'),
            "eps_forward": safe_get('forwardEps'),
            "revenue": safe_get('totalRevenue'),
            "revenue_formatted": format_large_number(safe_get('totalRevenue')),
            "profit_margin": safe_get('profitMargins'),
            "operating_margin": safe_get('operatingMargins'),
            "roe": safe_get('returnOnEquity'),
            
            # Trading
            "volume": safe_get('volume'),
            "avg_volume": safe_get('averageVolume'),
            "beta": safe_get('beta'),
            
            # Dividends
            "dividend_yield": safe_get('dividendYield'),
            "dividend_rate": safe_get('dividendRate'),
        }
    
    def _format_output(self, ticker: str, data: Dict[str, Any]) -> str:
        """Format extracted data into a structured string for LLM consumption."""
        
        def fmt(value: Any, prefix: str = "", suffix: str = "") -> str:
            """Format value with prefix/suffix or return N/A."""
            if value is None or value == "N/A":
                return "N/A"
            if isinstance(value, float):
                return f"{prefix}{value:.2f}{suffix}"
            return f"{prefix}{value}{suffix}"
        
        def fmt_pct(value: Any) -> str:
            """Format as percentage."""
            if value is None:
                return "N/A"
            return f"{value * 100:.2f}%"
        
        lines = [
            f"FINANCIAL DATA: {data['company_name']} ({ticker})",
            "=" * 60,
            "",
            "PRICE INFORMATION",
            "-" * 40,
            f"  Current Price:     {fmt(data['current_price'], '$')} {data['currency']}",
            f"  Previous Close:    {fmt(data['previous_close'], '$')}",
            f"  Day Range:         {fmt(data['day_low'], '$')} - {fmt(data['day_high'], '$')}",
            f"  52-Week Range:     {fmt(data['fifty_two_week_low'], '$')} - {fmt(data['fifty_two_week_high'], '$')}",
            "",
            "VALUATION METRICS",
            "-" * 40,
            f"  Market Cap:        {data['market_cap_formatted']}",
            f"  P/E (TTM):         {fmt(data['trailing_pe'])}",
            f"  P/E (Forward):     {fmt(data['forward_pe'])}",
            f"  Price/Book:        {fmt(data['price_to_book'])}",
            f"  PEG Ratio:         {fmt(data['peg_ratio'])}",
            f"  EV/EBITDA:         {fmt(data['ev_to_ebitda'])}",
            "",
            "FUNDAMENTALS",
            "-" * 40,
            f"  EPS (TTM):         {fmt(data['eps_ttm'], '$')}",
            f"  Revenue:           {data['revenue_formatted']}",
            f"  Profit Margin:     {fmt_pct(data['profit_margin'])}",
            f"  Operating Margin:  {fmt_pct(data['operating_margin'])}",
            f"  ROE:               {fmt_pct(data['roe'])}",
            "",
            "TRADING INFO",
            "-" * 40,
            f"  Volume:            {data['volume']:,}" if data['volume'] else "  Volume:            N/A",
            f"  Avg Volume:        {data['avg_volume']:,}" if data['avg_volume'] else "  Avg Volume:        N/A",
            f"  Beta:              {fmt(data['beta'])}",
            "",
            "COMPANY INFO",
            "-" * 40,
            f"  Sector:            {data['sector']}",
            f"  Industry:          {data['industry']}",
            "",
            f"Data retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ]
        
        return "\n".join(lines)
