"""
finresearch_advanced.py

Multi-agent financial research system for the
SDS-CP044 FinResearch AI - Advanced Track.

Usage (inside the repo, after `pip install -r requirements.txt`):

    export OPENAI_API_KEY="sk-..."   # or set in .env
    python finresearch_advanced.py --ticker AAPL --topic "long-term growth prospects"

This will:
  1. Use a Researcher agent to pull recent web/news context.
  2. Use a Financial Analyst agent to compute ratios & basic stats.
  3. Use a Reporting agent to draft an investment-style memo.
  4. Use a Manager agent to review and polish the final report.
"""

import os
import argparse
from typing import List

from dotenv import load_dotenv

import yfinance as yf
from duckduckgo_search import DDGS

import signal

# ---- Windows Workaround: Create any missing UNIX signals -------------
# CrewAI imports many UNIX-only signals (SIGHUP, SIGTSTP, SIGCONT, SIGUSR1, etc.)
# On Windows they don't exist, so we map all missing ones to SIGINT.
for name in dir(signal):
    pass  # just scanning existing names — we don't modify these

# Common UNIX signals CrewAI may reference:
unix_signals = [
    "SIGHUP", "SIGTERM", "SIGKILL", "SIGUSR1", "SIGUSR2",
    "SIGTSTP", "SIGSTOP", "SIGCONT", "SIGCHLD", "SIGTTIN",
    "SIGTTOU", "SIGQUIT"
]

for sig in unix_signals:
    if not hasattr(signal, sig):
        setattr(signal, sig, signal.SIGINT)
# ---------------------------------------------------------------------


from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ---------------------------------------------------------------------
#  Environment setup
# ---------------------------------------------------------------------

load_dotenv()  # loads OPENAI_API_KEY, etc., from .env if present


# ---------------------------------------------------------------------
#  Custom Tools
# ---------------------------------------------------------------------

@tool("web_search_finance")
def web_search_finance(query: str) -> str:
    """
    Search the web/news (general-purpose, free search) for finance-related queries
    and return a compact summary of the top results.
    """
    snippets = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=6):
            title = result.get("title", "")
            body = result.get("body", "")
            href = result.get("href", "")
            if title or body:
                snippets.append(f"{title}\n{body}\nURL: {href}")
    if not snippets:
        return "No relevant results found."
    return "\n\n---\n\n".join(snippets)


@tool("yfinance_fundamentals")
def yfinance_fundamentals(ticker: str) -> str:
    """
    Fetch basic fundamentals for a single equity ticker using yfinance.
    Returns key valuation and profitability metrics as text.
    """
    t = yf.Ticker(ticker)
    info = t.info if hasattr(t, "info") else {}

    # yfinance `.info` can be flaky; handle missing keys gracefully
    def get(key, default="N/A"):
        return info.get(key, default)

    lines = [
        f"Ticker: {ticker}",
        f"Long name: {get('longName')}",
        f"Sector: {get('sector')}",
        f"Industry: {get('industry')}",
        "",
        "Valuation:",
        f"  Market cap: {get('marketCap')}",
        f"  Trailing P/E: {get('trailingPE')}",
        f"  Forward P/E: {get('forwardPE')}",
        f"  PEG ratio: {get('pegRatio')}",
        "",
        "Profitability:",
        f"  Return on equity (ROE): {get('returnOnEquity')}",
        f"  Return on assets (ROA): {get('returnOnAssets')}",
        "",
        "Growth (if available):",
        f"  Revenue growth: {get('revenueGrowth')}",
        f"  Earnings growth: {get('earningsGrowth')}",
    ]
    return "\n".join(lines)


@tool("yfinance_price_history")
def yfinance_price_history(ticker: str) -> str:
    """
    Pull 1 year of daily price data and compute basic stats:
    average daily return, annualized volatility, and max drawdown.
    """
    import pandas as pd
    import numpy as np

    data = yf.Ticker(ticker).history(period="1y")
    if data.empty:
        return f"No price data available for {ticker}"

    data = data.dropna()
    data["return"] = data["Close"].pct_change()
    daily_ret = data["return"].dropna()

    avg_daily_ret = daily_ret.mean()
    vol_daily = daily_ret.std()
    # rough 252-trading-day scaling
    ann_ret = (1 + avg_daily_ret) ** 252 - 1
    ann_vol = vol_daily * (252 ** 0.5)

    # Max drawdown
    cum = (1 + daily_ret).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    summary = {
        "ticker": ticker,
        "start_date": str(data.index.min().date()),
        "end_date": str(data.index.max().date()),
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "max_drawdown": float(max_dd),
    }

    # Turn into readable text
    lines = [
        f"Price history stats for {ticker} (last 1y):",
        f"  Period: {summary['start_date']} → {summary['end_date']}",
        f"  Annualized return (approx): {summary['annualized_return']:.2%}",
        f"  Annualized volatility (approx): {summary['annualized_volatility']:.2%}",
        f"  Max drawdown: {summary['max_drawdown']:.2%}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
#  Agent Factory
# ---------------------------------------------------------------------

def build_agents() -> dict:
    """
    Create the four main agents used in the Advanced Track architecture:
      - manager
      - researcher
      - analyst
      - reporter

    We rely on the default LLM (OPENAI_MODEL_NAME or gpt-4o-mini).
    """
    manager = Agent(
        role="Portfolio Manager & Quality Controller",
        goal=(
            "Coordinate a team of specialist agents to produce a coherent, "
            "factually grounded, and decision-ready financial research report."
        ),
        backstory=(
            "You are a senior portfolio manager overseeing an AI research team. "
            "You review their work, ensure consistency, and give a clear final view."
        ),
        allow_delegation=False,
        verbose=True,
    )

    researcher = Agent(
        role="Financial News & Web Researcher",
        goal=(
            "Collect recent and relevant information about the company, sector, "
            "and macro environment from the web and news sources."
        ),
        backstory=(
            "You are an experienced equity research associate, excellent at "
            "finding concise, high-signal information online."
        ),
        tools=[web_search_finance],
        allow_delegation=False,
        verbose=True,
    )

    analyst = Agent(
        role="Fundamental & Quantitative Financial Analyst",
        goal=(
            "Use structured financial data to compute key ratios, describe "
            "business quality, growth, and risk/volatility in a clear way."
        ),
        backstory=(
            "You are a buy-side analyst with strong fundamentals and "
            "quant skills, comfortable with valuation and risk metrics."
        ),
        tools=[yfinance_fundamentals, yfinance_price_history],
        allow_delegation=False,
        verbose=True,
    )

    reporter = Agent(
        role="Investment Report Writer",
        goal=(
            "Turn all research and analysis into an investment-style memo "
            "with sections, bullets, and a clear bottom-line view."
        ),
        backstory=(
            "You previously worked at a major asset manager turning analysts' "
            "work into polished reports for investment committees."
        ),
        allow_delegation=False,
        verbose=True,
    )

    return {
        "manager": manager,
        "researcher": researcher,
        "analyst": analyst,
        "reporter": reporter,
    }


# ---------------------------------------------------------------------
#  Tasks & Crew
# ---------------------------------------------------------------------

def build_tasks(agents: dict) -> List[Task]:
    """
    Define the workflow as a sequence of Tasks.
    We still use multiple agents (Advanced Track idea), but orchestrate
    them with a sequential Process for simplicity.
    """
    manager = agents["manager"]
    researcher = agents["researcher"]
    analyst = agents["analyst"]
    reporter = agents["reporter"]

    research_task = Task(
        description=(
            "You are the news & web research specialist.\n\n"
            "User topic: {topic}\n"
            "Company ticker: {ticker}\n\n"
            "1. Use your web_search_finance tool to gather the most recent and "
            "relevant information about:\n"
            "   - The company and its latest developments\n"
            "   - Sector/industry trends\n"
            "   - Any major macro or regulatory news that could impact it.\n"
            "2. Focus on the last 12–18 months.\n"
            "3. Avoid duplicate headlines.\n\n"
            "Return 10–15 bullet points with short, factual summaries and URLs where possible."
        ),
        expected_output=(
            "A bullet list of 10–15 concise items covering company-specific, "
            "sector, and macro news with URLs where available."
        ),
        agent=researcher,
    )

    analysis_task = Task(
        description=(
            "You are the fundamental & quantitative analyst.\n\n"
            "Using the context from the research task plus your tools:\n"
            "- yfinance_fundamentals\n"
            "- yfinance_price_history\n\n"
            "For the company with ticker {ticker}, do the following:\n"
            "1. Summarize business model, main revenue drivers, and sector positioning.\n"
            "2. Compute and interpret key metrics (use the tool outputs as raw material):\n"
            "   - Market cap, trailing P/E, forward P/E, PEG.\n"
            "   - ROE, ROA, revenue / earnings growth (if available).\n"
            "   - Annualized return, volatility, and max drawdown for ~1 year.\n"
            "3. Highlight what these numbers say about valuation (cheap/fair/expensive), "
            "profitability, growth, and risk.\n\n"
            "Return your answer as clearly labeled sections (Valuation, Profitability, "
            "Growth, Risk) in markdown."
        ),
        expected_output=(
            "Markdown sections: Valuation, Profitability, Growth, Risk, each with "
            "bullet points that interpret the metrics in plain language."
        ),
        agent=analyst,
    )

    reporting_task = Task(
        description=(
            "You are the investment report writer.\n\n"
            "Using everything produced so far (news research + financial analysis), "
            "write a professional, human-readable memo for a portfolio manager.\n\n"
            "The question/topic from the user is:\n"
            "  {topic}\n\n"
            "Structure the report in markdown with these sections:\n"
            "1. Executive Summary (3–5 bullets with the key takeaway).\n"
            "2. Company Overview.\n"
            "3. Recent Developments & News.\n"
            "4. Fundamental Snapshot (valuation / profitability / growth).\n"
            "5. Risk Factors.\n"
            "6. Investment View (bull/bear case, and your balanced conclusion).\n\n"
            "Be specific but concise. If data is missing, say so explicitly."
        ),
        expected_output=(
            "A clean markdown investment memo with all the sections above, ready "
            "to paste into a research wiki or email."
        ),
        agent=reporter,
    )

    manager_qc_task = Task(
        description=(
            "You are the senior portfolio manager reviewing the draft memo.\n\n"
            "1. Read the entire context (research + analysis + draft report).\n"
            "2. Check that it directly addresses the user topic:\n"
            "      {topic}\n"
            "   for the company ticker:\n"
            "      {ticker}\n"
            "3. Fix any obvious inconsistencies, repetitions, or unclear wording.\n"
            "4. Add a very short 'Manager Verdict' at the top with:\n"
            "   - One-line overall stance (e.g. 'Cautious Buy', 'Hold', etc.)\n"
            "   - 2–3 bullets: key reasons and main risk.\n\n"
            "Return the final polished report (markdown only)."
        ),
        expected_output=(
            "A polished markdown report, starting with a 'Manager Verdict' section, "
            "followed by the improved memo."
        ),
        agent=manager,
    )

    return [research_task, analysis_task, reporting_task, manager_qc_task]


def build_finresearch_crew() -> Crew:
    """
    Assemble the full advanced-track crew with multiple agents and tasks.
    """
    agents = build_agents()
    tasks = build_tasks(agents)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,   # simple multi-agent pipeline
        verbose=True,
        output_log_file=False,        # set to True or a filename if you want logs
    )
    return crew


# ---------------------------------------------------------------------
#  CLI entrypoint
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the FinResearch AI multi-agent financial research system."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Equity ticker, e.g. AAPL, MSFT, TSLA",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Is this an attractive long-term investment?",
        help="User research question or angle (e.g. 'Is this stock overvalued?').",
    )

    args = parser.parse_args()

    crew = build_finresearch_crew()
    result = crew.kickoff(
        inputs={
            "ticker": args.ticker.upper(),
            "topic": args.topic,
        }
    )

    print("\n" + "=" * 80 + "\nFINAL REPORT\n" + "=" * 80 + "\n")
    print(result)


if __name__ == "__main__":
    main()
