import gradio as gr
import json
import re
from src.graph import create_graph

def parse_report_sections(report_text):
    """
    Parses the markdown report into dictionary sections for tabs.
    """
    sections = {
        "Executive Summary": "No summary available.",
        "Company Snapshot": "No snapshot available.",
        "Financial Indicators": "No data available.",
        "News & Sentiment": "No news available.",
        "Risks & Opportunities": "No analysis available.",
        "Final Perspective": "No conclusion available."
    }
    
    # Simple regex to capture content between headers
    patterns = {
        "Executive Summary": r"## 1\. Executive Summary\n(.*?)\n##",
        "Company Snapshot": r"## 2\. Company Snapshot\n(.*?)\n##",
        "Financial Indicators": r"## 3\. Key Financial Indicators\n(.*?)\n##",
        "News & Sentiment": r"## 4\. Recent News & Sentiment\n(.*?)\n##",
        "Risks & Opportunities": r"## 5\. Risks & Opportunities\n(.*?)\n##",
        "Final Perspective": r"## 6\. Final Perspective\n(.*?)$"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()
            
    return sections

def run_research(ticker, mode):
    """
    Main handler for Gradio. Runs the graph and returns outputs.
    """
    print(f"UI Request: Researching {ticker} in {mode} mode...")
    
    app = create_graph()
    initial_state = {
        "ticker": ticker,
        "investor_mode": mode,
        "messages": [],
        "research_summary": "",
        "financial_data": {},
        "final_report": "",
        "next_step": ""
    }
    
    final_output = app.invoke(initial_state)
    report = final_output.get("final_report", "Error generating report.")
    
    # Parse sections
    sections = parse_report_sections(report)
    
    # Prepare files for download
    md_filename = f"{ticker}_report.md"
    json_filename = f"{ticker}_report.json"
    
    with open(md_filename, "w") as f:
        f.write(report)
        
    with open(json_filename, "w") as f:
        json.dump(final_output.get("financial_data", {}), f, indent=2)
        
    return (
        report, 
        sections["Executive Summary"],
        sections["Company Snapshot"],
        sections["Financial Indicators"],
        sections["News & Sentiment"],
        sections["Risks & Opportunities"],
        md_filename,
        json_filename
    )

# Build UI
with gr.Blocks(title="FinResearch AI") as demo:
    gr.Markdown("# 📈 FinResearch AI Agent Team")
    
    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(label="Stock Ticker", placeholder="e.g. NVDA")
            mode_input = gr.Dropdown(
                choices=["Neutral", "Bullish", "Bearish"], 
                value="Neutral", 
                label="Investor Persona"
            )
            submit_btn = gr.Button("🚀 Run Research", variant="primary")
            
            with gr.Row():
                md_file = gr.File(label="Download Markdown")
                json_file = gr.File(label="Download Financial Data")
        
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("📊 Full Report"):
                    report_output = gr.Markdown()
                with gr.TabItem("📝 Executive Summary"):
                    summary_output = gr.Markdown()
                with gr.TabItem("🏢 Company Snapshot"):
                    snapshot_output = gr.Markdown()
                with gr.TabItem("📈 Financials"):
                    financials_output = gr.Markdown()
                with gr.TabItem("🗞️ News"):
                    news_output = gr.Markdown()
                with gr.TabItem("⚖️ Risks/Opps"):
                    risks_output = gr.Markdown()

    submit_btn.click(
        fn=run_research,
        inputs=[ticker_input, mode_input],
        outputs=[
            report_output, 
            summary_output, 
            snapshot_output, 
            financials_output, 
            news_output, 
            risks_output,
            md_file,
            json_file
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
