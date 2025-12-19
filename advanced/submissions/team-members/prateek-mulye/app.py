"""
FinResearch AI - Advanced Multi-Agent Financial Research System
Built with LangGraph, LangChain, and Gradio.

This application acts as a main entry point for the user interface.
"""

import time
import json
import re
import gradio as gr
import yfinance as yf
from src.graph import create_graph
import plotly.graph_objects as go

# --- Security & Caching ---
CACHE_DURATION = 300  # 5 minutes
RATE_LIMIT_BLOCK = 1800  # 30 minutes
MAX_REQUESTS_PER_WINDOW = 5  # 5 requests before block
RATE_LIMIT_WINDOW = 3600 # 1 hour

# Simple in-memory storage
# Cache key: (ticker, mode) -> (timestamp, data)
response_cache = {}
# Rate Limit: {client_ip: {"count": int, "window_start": timestamp, "blocked_until": timestamp}}
rate_limit_db = {}

# --- Funny Loading Experience ---
FUNNY_PHRASES = [
    "Jerome Powell is typing... stay calm...",
    "Convincing the algorithms to ignore the volatility...",
    "Mining for 'hopium' in the technical indicators...",
    "Teaching the bear to share its lunch with the bull...",
    "Simulating a soft landing (it's bumpy)...",
    "Asking the magic 8-ball about the P/E ratio...",
    "Hiding your portfolio from the margin callers...",
    "Calibrating the 'Moon' trajectory...",
    "Scanning the blockchain for missing 'Alpha'...",
    "Polishing the crystal ball for the analyst node..."
]

PULSE_CSS = """
@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}
.pulse-text {
    animation: pulse 2s infinite ease-in-out;
    color: #ff9800;
    font-weight: bold;
    font-style: italic;
    font-family: 'Roboto Mono', monospace;
    text-align: center;
    padding: 10px;
}
"""

def get_funny_loading_html():
    import random
    phrase = random.choice(FUNNY_PHRASES)
    return f'<div class="pulse-text">🚀 {phrase}</div>'

def get_client_ip(request: gr.Request):
    if request:
        return request.client.host
    return "unknown"

def check_rate_limit(request: gr.Request):
    """
    Blocks client if they exceed MAX_REQUESTS_PER_WINDOW.
    """
    client_ip = get_client_ip(request)
    now = time.time()
    
    if client_ip not in rate_limit_db:
        rate_limit_db[client_ip] = {"count": 0, "window_start": now, "blocked_until": 0}
        
    user_data = rate_limit_db[client_ip]
    
    if user_data["blocked_until"] > now:
        remaining = int((user_data["blocked_until"] - now) / 60)
        raise gr.Error(f"Rate limit exceeded. Try again in {remaining} minutes.")
        
    if now - user_data["window_start"] > RATE_LIMIT_WINDOW:
        user_data["count"] = 0
        user_data["window_start"] = now
        
    user_data["count"] += 1
    if user_data["count"] > MAX_REQUESTS_PER_WINDOW:
        user_data["blocked_until"] = now + RATE_LIMIT_BLOCK
        raise gr.Error("Rate limit exceeded. You are blocked for 30 minutes.")

def parse_report_sections(report_text):
    """
    Parses the markdown report into dictionary sections for tabs.
    """
    sections = {
        "Executive Summary": "No summary available.",
        "Analyst Verdict": "No verdict available.",
        "Company Snapshot": "No snapshot available.",
        "Financial Indicators": "No data available.",
        "News & Sentiment": "No news available.",
        "Risks & Opportunities": "No analysis available.",
        "Final Perspective": "No conclusion available."
    }
    
    patterns = {
        "Executive Summary": r"## 1\. Executive Summary\n(.*?)\n##",
        "Analyst Verdict": r"## 2\. Analyst Verdict\n(.*?)\n##",
        "Company Snapshot": r"## 3\. Company Snapshot\n(.*?)\n##",
        "Financial Indicators": r"## 4\. Key Financial Indicators\n(.*?)\n##",
        "News & Sentiment": r"## 6\. Recent News & Sentiment\n(.*?)\n##",
        "Risks & Opportunities": r"## 7\. Risks & Opportunities\n(.*?)\n##",
        "Final Perspective": r"## 8\. Final Perspective\n(.*?)$"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()
            
    return sections

def run_research(ticker, mode, request: gr.Request):
    """
    Main handler for Gradio. Runs the graph and returns outputs.
    """
    check_rate_limit(request) # Security Check
    
    ticker = ticker.upper().strip()
    
    # Validation Guardrail
    if not ticker:
         raise gr.Error("Please enter a ticker symbol.")
         
    try:
        tick = yf.Ticker(ticker)
        # Fast check: history for 1 day
        hist = tick.history(period="1d")
        if hist.empty:
             raise ValueError("No data found")
    except Exception:
        raise gr.Error(f"Invalid Ticker: {ticker}. Please enter a valid symbol.")

    cache_key = (ticker, mode)
    now = time.time()
    
    # Check Cache
    if cache_key in response_cache:
        timestamp, cached_output = response_cache[cache_key]
        if now - timestamp < CACHE_DURATION:
            print(f"Returning cached response for {ticker}...")
            final_output = cached_output
        else:
            print(f"Cache expired for {ticker}. Fetching new data...")
            app = create_graph()
            initial_state = {"ticker": ticker, "investor_mode": mode, "messages": []}
            final_output = app.invoke(initial_state)
            response_cache[cache_key] = (now, final_output)
    else:
        print(f"Fetching fresh data for {ticker}...")
        app = create_graph()
        initial_state = {"ticker": ticker, "investor_mode": mode, "messages": []}
        final_output = app.invoke(initial_state)
        response_cache[cache_key] = (now, final_output)
        
    report = final_output.get("final_report", "Error generating report.")
    sections = parse_report_sections(report)
    
    md_filename = f"{ticker}_report.md"
    json_filename = f"{ticker}_report.json"
    
    financial_data = final_output.get("financial_data", {})
    analyst_verdict = final_output.get("analyst_verdict", {})
    
    plot_update = gr.Plot(visible=False)
    metrics_update = gr.Plot(visible=False)
    verdict_html = "<div style='color: #888;'>Research not started.</div>"
    
    # 1. Process Chart (Radar)
    if isinstance(financial_data, dict):
        chart_data = financial_data.get("chart", {})
        if chart_data.get("type") == "radar":
            try:
                data_points = chart_data.get("data", {})
                categories = list(data_points.keys())
                values = list(data_points.values())
                
                # Close the loop
                categories = [*categories, categories[0]]
                values = [*values, values[0]]
                
                if mode != "Bearish":
                   line_color = '#00C9FF' # Cyan
                   fill_color = 'rgba(0, 201, 255, 0.3)'
                else:
                   line_color = '#FF5252' # Red
                   fill_color = 'rgba(255, 82, 82, 0.3)'

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=ticker,
                    line_color=line_color,
                    fillcolor=fill_color
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10], showline=False, gridcolor="#444444"),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    title=dict(text=""),
                    margin=dict(t=20, b=20, l=40, r=40),
                    height=350,
                    font=dict(color="white")
                )
                plot_update = gr.Plot(value=fig, visible=True)
                
            except Exception as e:
                print(f"Radar plot generation failed: {e}")

        # 2. Process Metrics (Indicator Grid)
        metrics_list = financial_data.get("metrics", [])
        if metrics_list:
            top_metrics = metrics_list[:4]
            fig_metrics = go.Figure()
            fig_metrics.update_layout(template="plotly_dark")
            
            positions = [
                (0.25, 0.80), (0.75, 0.80),
                (0.25, 0.30), (0.75, 0.30)
            ]
            
            for i, m in enumerate(top_metrics):
                if i >= 4: break
                val = m.get("Value", "-")
                label = m.get("Metric", "Metric")
                x_pos, y_pos = positions[i]
                
                # Value
                fig_metrics.add_annotation(
                    x=x_pos, y=y_pos,
                    text=str(val),
                    showarrow=False,
                    font=dict(size=30, color="#00C9FF", family="Roboto Mono, monospace"),
                    xref="paper", yref="paper"
                )
                # Label
                fig_metrics.add_annotation(
                    x=x_pos, y=y_pos - 0.15,
                    text=label.upper(),
                    showarrow=False,
                    font=dict(size=12, color="#E0E0E0", family="Roboto"),
                    xref="paper", yref="paper"
                )

            fig_metrics.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=350,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis={'visible': False, 'showgrid': False, 'zeroline': False},
                yaxis={'visible': False, 'showgrid': False, 'zeroline': False},
                title=dict(text="")
            )
            metrics_update = gr.Plot(value=fig_metrics, visible=True)

    # 3. Process Verdict (HTML Card)
    if isinstance(analyst_verdict, dict):
        score = analyst_verdict.get("score", 0)
        rec = analyst_verdict.get("recommendation", "N/A")
        reason = analyst_verdict.get("reasoning", "No reasoning provided.")
        
        if score >= 75:
            score_color = "#00E676" # Green
            badge_bg = "rgba(0, 230, 118, 0.15)"
            border_color = "rgba(0, 230, 118, 0.4)"
        elif score >= 40:
            score_color = "#FFD700" # Gold
            badge_bg = "rgba(255, 215, 0, 0.15)"
            border_color = "rgba(255, 215, 0, 0.4)"
        else:
            score_color = "#FF5252" # Red
            badge_bg = "rgba(255, 82, 82, 0.15)"
            border_color = "rgba(255, 82, 82, 0.4)"
            
        verdict_html = f"""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%); border-radius: 16px; padding: 25px; display: flex; align-items: center; justify-content: space-between; gap: 30px; border: 1px solid {border_color}; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
            <div style="flex: 1; text-align: center; border-right: 1px solid rgba(255,255,255,0.08); padding-right: 30px;">
                <div style="font-size: 13px; color: #aaa; letter-spacing: 1px; margin-bottom: 5px;">ANALYST SCORE</div>
                <div style="font-size: 64px; font-weight: 800; color: {score_color}; line-height: 1; text-shadow: 0 0 15px {badge_bg};">{score}</div>
                <div style="font-size: 11px; color: #666; margin-top: 5px;">OUT OF 100</div>
            </div>
            
            <div style="flex: 1; text-align: center; border-right: 1px solid rgba(255,255,255,0.08); padding-right: 30px;">
                <div style="font-size: 13px; color: #aaa; letter-spacing: 1px; margin-bottom: 12px;">VERDICT</div>
                <div style="background-color: {badge_bg}; color: {score_color}; padding: 10px 24px; border-radius: 30px; font-weight: 800; font-size: 20px; border: 1px solid {border_color}; display: inline-block;">
                    {rec.upper()}
                </div>
            </div>
            
            <div style="flex: 2.5; padding-left: 10px;">
                 <div style="font-size: 13px; color: #aaa; letter-spacing: 1px; margin-bottom: 10px;">KEY REASONING</div>
                 <div style="font-size: 16px; color: #f0f0f0; line-height: 1.6; font-weight: 300; border-left: 3px solid {score_color}; padding-left: 15px;">
                    "{reason}"
                 </div>
            </div>
        </div>
        """

    with open(md_filename, "w") as f:
        f.write(report)
        
    with open(json_filename, "w") as f:
        json.dump({"financial_data": financial_data, "verdict": analyst_verdict}, f, indent=2)
        
    return (
        report, 
        sections["Executive Summary"],
        sections["Company Snapshot"],
        sections["News & Sentiment"],
        sections["Risks & Opportunities"],
        md_filename,
        json_filename,
        plot_update,
        metrics_update,
        verdict_html
    )

# --- UI Layout ---
with gr.Blocks(title="FinResearch AI", theme=gr.themes.Soft(), css=PULSE_CSS + " footer {visibility: hidden}") as demo:
    gr.HTML("""
    <div style="background-color: #f0ad4e; color: white; display: flex; align-items: center; justify-content: center; padding: 10px; font-weight: bold; font-family: sans-serif; width: 100%; border-bottom: 2px solid #eea236;">
        <span style="font-size: 20px; margin-right: 10px;">⚠️</span>
        <span>DISCLAIMER: This tool uses LLMs for financial analysis. It is NON-ADVISORY and for INFORMATIONAL PURPOSES ONLY. Investing involves risk. Do your own due diligence.</span>
    </div>
    """)
    
    gr.Markdown("# 🤖 FinResearch AI: Intelligent Stock Analysis")
    gr.Markdown("An advanced multi-agent system for real-time financial research, analysis, and visualization.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                ticker_input = gr.Textbox(label="Stock Ticker", placeholder="e.g. MSFT", elem_id="ticker-input")
                mode_input = gr.Dropdown(
                    choices=["Neutral", "Bullish", "Bearish"], 
                    value="Neutral", 
                    label="Investor Persona"
                )
                submit_btn = gr.Button("🚀 Run Research", variant="primary")
                
                # Funny Loading Status
                loading_display = gr.HTML(visible=True)
            
            gr.Markdown("### 📥 Downloads")
            with gr.Row():
                md_file = gr.File(label="Report (MD)")
                json_file = gr.File(label="Data (JSON)")
        
        with gr.Column(scale=4):
            # Verdict Card
            verdict_display = gr.HTML(
                value="<div style='padding: 20px; text-align: center; color: #888;'>Ready to analyze...</div>",
                label="Analyst Verdict",
            )
            
            # Data & Visuals
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    metrics_table = gr.Plot(label="Key Metrics Grid", visible=False)
                with gr.Column(scale=1):
                    viz_plot = gr.Plot(label="Financial Health Radar", visible=False)
                    gr.Markdown("ℹ️ *Visualizing Key Metrics vs Industry or Trends. Source: YFinance/TradingView.*")
            
            # Detailed Reports
            with gr.Tabs():
                with gr.TabItem("📝 Executive Summary"):
                    summary_output = gr.Markdown()
                with gr.TabItem("🏢 Company Snapshot"):
                    snapshot_output = gr.Markdown()
                with gr.TabItem("🗞️ News & Sentiment"):
                    news_output = gr.Markdown()
                with gr.TabItem("⚖️ Risks & Opportunities"):
                    risks_output = gr.Markdown()
                with gr.TabItem("📄 Full Report"):
                    report_output = gr.Markdown()    

    # Event Chaining for Funny Loading Experience
    submit_btn.click(
        fn=get_funny_loading_html,
        outputs=loading_display
    ).then(
        fn=run_research,
        inputs=[ticker_input, mode_input],
        outputs=[
            report_output, summary_output, snapshot_output, news_output, risks_output,
            md_file, json_file, viz_plot, metrics_table, verdict_display
        ]
    ).then(
        fn=lambda: "", # Clear loading message after completion
        outputs=loading_display
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
