import os
import sys
import argparse
from dotenv import load_dotenv

# Add src to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import create_graph
from src.memory import VectorMemory

load_dotenv()

def test_flow(ticker="MSFT"):
    print(f"--- Running Verification Test for {ticker} ---")
    
    # 1. Run Graph
    app = create_graph()
    initial_state = {
        "ticker": ticker,
        "messages": [],
        "research_summary": "",
        "financial_data": {},
        "next_step": ""
    }
    
    print("Invoking Graph...")
    # Using invoke instead of stream for test simplicity, but loop is fine
    final_output = app.invoke(initial_state)
    
    print("\n--- Graph Execution Completed ---")
    
    # 2. Verify Pinecone Content
    memory = VectorMemory()
    print("\nVerifying Pinecone Content...")
    
    # Search for any documents related to the ticker
    # We can try a specific query to see if data retrieval works
    query = f"financial info for {ticker}"
    
    # Test Similarity Search
    results = memory.similarity_search(query, k=5)
    
    found_news = False
    found_financials = False
    
    print(f"Query Results for '{query}':")
    for doc in results:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        # Check metadata match (heuristic, since search is semantic)
        # Note: metadata values might be lists or strings depending on ingestion
        doc_ticker = meta.get("ticker", "")
        
        print(f"- [Source: {source}] Text snippet: {doc.page_content[:100]}...")
        
        if source == "researcher_agent_summary" or source == "tavily_search":
            found_news = True
        if source == "analyst_agent" or source == "yfinance":
            found_financials = True
            
    if found_news:
        print("✅ Found News/Summary Documents")
    else:
        print("⚠️ No News/Summary Documents found in top 5 results.")
        
    if found_financials:
        print("✅ Found Financial Analysis Documents")
    else:
        print("⚠️ No Financial Analysis Documents found in top 5 results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="MSFT")
    args = parser.parse_args()
    test_flow(args.ticker)
