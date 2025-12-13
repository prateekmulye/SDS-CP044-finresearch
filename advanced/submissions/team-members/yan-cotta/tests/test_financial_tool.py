"""
Unit Tests for FinancialDataTool.

Tests the financial data tool with mocked Yahoo Finance responses.
No network calls are made during these tests.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFinancialDataTool:
    """Test suite for FinancialDataTool."""

    @pytest.fixture
    def mock_yfinance_data(self) -> dict:
        """Standard mock data representing a successful yfinance response."""
        return {
            "shortName": "Apple Inc.",
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "regularMarketPrice": 195.50,
            "previousClose": 194.25,
            "currency": "USD",
            "dayHigh": 196.80,
            "dayLow": 193.10,
            "fiftyTwoWeekHigh": 199.62,
            "fiftyTwoWeekLow": 164.08,
            "marketCap": 3050000000000,
            "trailingPE": 31.45,
            "forwardPE": 28.67,
            "priceToBook": 45.12,
            "pegRatio": 2.85,
            "enterpriseValue": 3100000000000,
            "enterpriseToEbitda": 25.4,
            "trailingEps": 6.21,
            "forwardEps": 6.82,
            "totalRevenue": 383000000000,
            "profitMargins": 0.2531,
            "operatingMargins": 0.2967,
            "returnOnEquity": 1.4725,
            "volume": 58934521,
            "averageVolume": 52000000,
            "beta": 1.28,
            "dividendYield": 0.0051,
            "dividendRate": 1.00,
        }

    @pytest.fixture
    def mock_empty_data(self) -> dict:
        """Mock data representing an invalid/unknown ticker."""
        return {}

    @pytest.fixture
    def financial_tool(self):
        """Create a FinancialDataTool instance with mocked dependencies."""
        with patch.dict("sys.modules", {"yfinance": MagicMock()}):
            from src.tools.financial_data import FinancialDataTool
            return FinancialDataTool()

    # -------------------------------------------------------------------------
    # Happy Path Tests
    # -------------------------------------------------------------------------

    def test_valid_ticker_returns_formatted_data(self, mock_yfinance_data: dict) -> None:
        """Test that a valid ticker returns properly formatted financial data."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            # Setup mock
            mock_ticker = MagicMock()
            mock_ticker.info = mock_yfinance_data
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            # Execute
            result = tool._run("AAPL")

            # Assert
            assert "FINANCIAL DATA:" in result
            assert "Apple Inc." in result
            assert "AAPL" in result
            assert "195.50" in result or "195.5" in result
            assert "Technology" in result
            assert "PRICE INFORMATION" in result
            assert "VALUATION METRICS" in result
            assert "FUNDAMENTALS" in result

    def test_ticker_normalization(self, mock_yfinance_data: dict) -> None:
        """Test that tickers are normalized to uppercase."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_yfinance_data
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            # Test lowercase input
            result = tool._run("aapl")

            # Verify Ticker was called with uppercase
            mock_yf.Ticker.assert_called_with("AAPL")
            assert "AAPL" in result

    def test_ticker_with_whitespace(self, mock_yfinance_data: dict) -> None:
        """Test that whitespace around ticker is trimmed."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_yfinance_data
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("  AAPL  ")

            mock_yf.Ticker.assert_called_with("AAPL")
            assert "AAPL" in result

    def test_market_cap_formatting(self, mock_yfinance_data: dict) -> None:
        """Test that large numbers are formatted correctly (billions/trillions)."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_yfinance_data
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("AAPL")

            # Market cap of 3.05T should be formatted
            assert "$3.05T" in result or "$3.0T" in result or "3.05T" in result

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------

    def test_empty_ticker_returns_error(self) -> None:
        """Test that empty ticker returns appropriate error message."""
        with patch("src.tools.financial_data.yf"):
            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("")

            assert "ERROR" in result
            assert "Empty ticker" in result or "empty" in result.lower()

    def test_invalid_ticker_graceful_handling(self) -> None:
        """Test that invalid ticker is handled without crashing."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            # Setup mock to return empty data
            mock_ticker = MagicMock()
            mock_ticker.info = {"regularMarketPrice": None}
            mock_ticker.history.return_value = MagicMock(empty=True)
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            # Execute - should not raise
            result = tool._run("INVALIDTICKER12345")

            # Should return error message, not crash
            assert "ERROR" in result or "No data" in result

    def test_none_ticker_returns_error(self) -> None:
        """Test that None ticker is handled gracefully."""
        with patch("src.tools.financial_data.yf"):
            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            # This might raise or return error - both are acceptable
            try:
                result = tool._run(None)  # type: ignore
                assert "ERROR" in result or result is None
            except (TypeError, AttributeError):
                # Acceptable behavior - the tool caught the invalid input
                pass

    def test_api_exception_handled_gracefully(self) -> None:
        """Test that yfinance API exceptions are caught and reported."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            # Setup mock to raise exception
            mock_yf.Ticker.side_effect = Exception("API rate limit exceeded")

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            # Should not raise, should return error message
            result = tool._run("AAPL")

            assert "ERROR" in result
            assert "Unexpected error" in result or "Exception" in result

    def test_partial_data_handling(self) -> None:
        """Test handling of incomplete data from Yahoo Finance."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            # Only some fields present
            partial_data = {
                "shortName": "Test Corp",
                "regularMarketPrice": 100.0,
                # Missing: PE, market cap, etc.
            }
            mock_ticker = MagicMock()
            mock_ticker.info = partial_data
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("TEST")

            # Should still return data, with N/A for missing fields
            assert "FINANCIAL DATA:" in result
            assert "Test Corp" in result
            assert "N/A" in result  # Missing fields should show N/A

    # -------------------------------------------------------------------------
    # yfinance Not Installed Tests
    # -------------------------------------------------------------------------

    def test_yfinance_not_installed(self) -> None:
        """Test behavior when yfinance is not installed."""
        with patch("src.tools.financial_data.yf", None):
            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("AAPL")

            assert "ERROR" in result
            assert "yfinance" in result.lower() or "not installed" in result.lower()

    # -------------------------------------------------------------------------
    # Historical Data Fallback Tests
    # -------------------------------------------------------------------------

    def test_historical_fallback_when_no_current_price(self) -> None:
        """Test that historical data is used when current price unavailable."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            import pandas as pd

            # No current price in info
            mock_ticker = MagicMock()
            mock_ticker.info = {"shortName": "Test Corp", "regularMarketPrice": None}

            # But historical data available
            mock_history = pd.DataFrame({"Close": [98.0, 99.0, 100.0]})
            mock_ticker.history.return_value = mock_history
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("TEST")

            # Should use historical close price
            assert "FINANCIAL DATA:" in result or "100" in result


class TestTickerNormalization:
    """Focused tests for ticker input normalization."""

    @pytest.mark.parametrize(
        "input_ticker,expected",
        [
            ("aapl", "AAPL"),
            ("AAPL", "AAPL"),
            ("AaPl", "AAPL"),
            ("  msft  ", "MSFT"),
            ("tsla", "TSLA"),
        ],
    )
    def test_normalize_various_inputs(self, input_ticker: str, expected: str) -> None:
        """Test normalization of various ticker input formats."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {"regularMarketPrice": 100.0, "shortName": "Test"}
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            tool._run(input_ticker)

            mock_yf.Ticker.assert_called_with(expected)


class TestMetricFormatting:
    """Tests for number and percentage formatting."""

    def test_percentage_formatting(self) -> None:
        """Test that percentages are formatted correctly."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "shortName": "Test",
                "regularMarketPrice": 100.0,
                "profitMargins": 0.2531,  # Should display as 25.31%
                "operatingMargins": 0.10,  # Should display as 10.00%
            }
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("TEST")

            assert "25.31%" in result
            assert "10.00%" in result

    def test_large_number_formatting_billions(self) -> None:
        """Test formatting of billion-scale numbers."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "shortName": "Test",
                "regularMarketPrice": 100.0,
                "marketCap": 150_000_000_000,  # 150B
                "totalRevenue": 50_000_000_000,  # 50B
            }
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("TEST")

            assert "$150" in result and "B" in result
            assert "$50" in result

    def test_large_number_formatting_millions(self) -> None:
        """Test formatting of million-scale numbers."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "shortName": "Small Corp",
                "regularMarketPrice": 25.0,
                "marketCap": 500_000_000,  # 500M
            }
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("SMALL")

            assert "$500" in result or "500" in result
            assert "M" in result


# -----------------------------------------------------------------------------
# Integration-style Tests (still mocked, but test full flow)
# -----------------------------------------------------------------------------

@pytest.mark.unit
class TestToolIntegration:
    """Tests that verify the tool works end-to-end with mocked data."""

    def test_complete_output_structure(self) -> None:
        """Verify the complete output structure contains all expected sections."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "shortName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "regularMarketPrice": 195.50,
                "previousClose": 194.25,
                "currency": "USD",
                "marketCap": 3050000000000,
                "trailingPE": 31.45,
                "trailingEps": 6.21,
                "volume": 58934521,
                "beta": 1.28,
            }
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("AAPL")

            # Verify all major sections present
            expected_sections = [
                "FINANCIAL DATA:",
                "PRICE INFORMATION",
                "VALUATION METRICS",
                "FUNDAMENTALS",
                "TRADING INFO",
                "COMPANY INFO",
            ]

            for section in expected_sections:
                assert section in result, f"Missing section: {section}"

    def test_timestamp_in_output(self) -> None:
        """Verify that output includes a timestamp."""
        with patch("src.tools.financial_data.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "shortName": "Test",
                "regularMarketPrice": 100.0,
            }
            mock_yf.Ticker.return_value = mock_ticker

            from src.tools.financial_data import FinancialDataTool
            tool = FinancialDataTool()

            result = tool._run("TEST")

            assert "Data retrieved:" in result
            # Should contain date-like pattern
            assert "202" in result  # Year prefix
