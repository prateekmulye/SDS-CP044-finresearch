import os
import textwrap

import streamlit as st

# Reuse everything you already have in finresearch_advanced.py
from finresearch_advanced import build_finresearch_crew


def run_research(ticker: str, topic: str) -> str:
    """Helper that runs the CrewAI pipeline and returns the final report."""
    crew = build_finresearch_crew()
    result = crew.kickoff(
        inputs={
            "ticker": ticker.upper(),
            "topic": topic,
        }
    )
    # CrewAI often returns an object with .raw or .output; str() is safest
    return str(result)


def main():
    st.set_page_config(
        page_title="FinResearch AI – Multi-Agent Financial Research",
        page_icon="📊",
        layout="wide",
    )

    # ------------- SIDEBAR -------------
    with st.sidebar:
        st.header("⚙️ Settings")

        st.markdown(
            "Set your OpenAI API key here or via the "
            "`OPENAI_API_KEY` environment variable."
        )

        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )

        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

        st.markdown("---")

        ticker = st.text_input(
            "Ticker symbol",
            value="AAPL",
            help="Example: AAPL, MSFT, TSLA, NVDA...",
        )

        topic = st.text_area(
            "Research question / angle",
            value="Is this an attractive long-term investment?",
            help="What do you want the agents to investigate?",
            height=100,
        )

        run_button = st.button("🚀 Run Multi-Agent Analysis", use_container_width=True)

    # ------------- MAIN PAGE -------------
    st.title("📊 FinResearch AI – Multi-Agent Financial Research System")

    st.markdown(
        textwrap.dedent(
            """
            This app uses a **multi-agent CrewAI system**:

            1. 📰 **Researcher** – collects recent web/news context.  
            2. 📈 **Analyst** – pulls fundamentals & price history and computes key ratios.  
            3. 📝 **Reporter** – writes a structured investment memo in markdown.  
            4. ✅ **Manager** – reviews the memo and adds a *Manager Verdict*.

            Enter a ticker and a question in the sidebar and click **Run Multi-Agent Analysis**.
            """
        )
    )

    if run_button:
        # Basic validation
        if not ticker.strip():
            st.error("Please enter a ticker (e.g. AAPL).")
            return

        if not (os.getenv("OPENAI_API_KEY") or api_key_input):
            st.error(
                "No OpenAI API key found. "
                "Set it in the sidebar or as OPENAI_API_KEY in your environment."
            )
            return

        st.info(
            f"Running analysis for **{ticker.upper()}** "
            f"with topic: *{topic.strip() or 'Is this an attractive long-term investment?'}*"
        )

        with st.spinner("Agents are working… this can take a little while ⏳"):
            try:
                report = run_research(ticker, topic)
                st.session_state["finresearch_report"] = report
            except Exception as e:
                st.error("Something went wrong while running the agents.")
                st.exception(e)
                return

    # Show last report if available
    if "finresearch_report" in st.session_state:
        st.markdown("---")
        st.subheader("📄 Final Report")
        st.markdown(st.session_state["finresearch_report"], unsafe_allow_html=False)
    else:
        st.markdown("---")
        st.info("No report yet. Configure the inputs in the sidebar and run the analysis.")


if __name__ == "__main__":
    main()
