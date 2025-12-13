from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from ..memory import VectorMemory
from ..state import AgentState

class ResearcherAgent:
    def __init__(self):
        self.search_tool = TavilySearch(max_results=5)
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-5-nano")

    def run(self, state: AgentState):
        """
        Researcher Agent Node logic.
        """
        print("--- Researcher Agent ---")
        ticker = state.get("ticker")
        
        # 1. Search
        query = f"Financial news and market sentiment for {ticker} stock"
        print(f"Searching for: {query}")
        
        results = self.search_tool.invoke(query)
        
        # DEBUG: Check type of results
        print(f"DEBUG: Results type: {type(results)}")
        print(f"DEBUG: Results content: {results}")

        if isinstance(results, str):
            try:
                # Try to parse if it's a JSON string
                parsed = json.loads(results)
                if isinstance(parsed, list):
                    results = parsed
                else:
                    # If JSON but not list (e.g. dict), or just string text
                    results = [{"url": "tavily_search", "content": results}]
            except:
                # If not JSON, it's just text answer
                results = [{"url": "tavily_search", "content": results}]
        elif isinstance(results, dict):
             # Some tools return a dict with keys like 'results' or 'answer'
             if "results" in results:
                 results = results["results"]
             else:
                 results = [{"url": "tavily_search", "content": str(results)}]
                 
        # 2. LLM Summarization
        # We want to synthesize the search results before storing
        raw_content = ""
        for res in results:
            # Ensure elements are dicts
            if isinstance(res, dict):
                raw_content += f"Source: {res.get('url', 'unknown')}\nContent: {res.get('content', '')}\n\n"
            else:
                raw_content += f"Content: {str(res)}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a specialized financial researcher. Summarize the following search results for {ticker} into a concise market update. Focus on sentiment, key events, and potential risks/opportunities."),
            ("user", "Search Results:\n{raw_content}")
        ])
        
        chain = prompt | self.llm
        print("Synthesizing research with LLM...")
        summary = chain.invoke({"ticker": ticker, "raw_content": raw_content})
        summary_text = summary.content
        
        # 3. Store in Pinecone
        # We store the synthesized summary as a high-value document
        doc = Document(
            page_content=summary_text, 
            metadata={
                "ticker": ticker, 
                "source": "researcher_agent_summary",
                "type": "market_summary"
            }
        )
        
        # We can also store the raw chunks if needed, but for now let's prioritize the insight
        self.memory.add_documents([doc], source="researcher_agent")
        print(f"Stored synthesized research summary in Pinecone.")
            
        # 4. Update State
        return {
            "research_summary": summary_text,
            "messages": [HumanMessage(content=f"Research completed for {ticker}.")]
        }

def researcher_node(state: AgentState):
    agent = ResearcherAgent()
    return agent.run(state)
