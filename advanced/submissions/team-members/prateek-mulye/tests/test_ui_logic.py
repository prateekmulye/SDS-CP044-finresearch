import sys
import os
import shutil

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import run_research

def test_ui_logic():
    print("--- Testing UI Logic Headless ---")
    ticker = "MSFT"
    mode = "Bullish"
    
    # Run logic
    print(f"Invoking run_research({ticker}, {mode})...")
    outputs = run_research(ticker, mode)
    
    # Unpack outputs
    (report, summary, snapshot, financials, news, risks, md_file, json_file) = outputs
    
    print("\n--- Verification ---")
    
    # 1. Check Report Content
    if len(report) > 100:
        print("✅ Report generated (Length > 100)")
    else:
        print("❌ Report generation failed or empty")
        
    # 2. Check Parsing (Sections)
    if summary != "No summary available.":
        print("✅ 'Executive Summary' parsed")
    else:
        print("⚠️ 'Executive Summary' NOT parsed")
        
    if risks != "No analysis available.":
        print("✅ 'Risks & Opportunities' parsed")
    else:
        print("⚠️ 'Risks & Opportunities' NOT parsed")
        
    # 3. Check Files
    if os.path.exists(md_file):
        print(f"✅ Markdown file created: {md_file}")
        # clean up
        os.remove(md_file)
    else:
        print(f"❌ Markdown file missing: {md_file}")
        
    if os.path.exists(json_file):
        print(f"✅ JSON file created: {json_file}")
        # clean up
        os.remove(json_file)
    else:
        print(f"❌ JSON file missing: {json_file}")

if __name__ == "__main__":
    test_ui_logic()
