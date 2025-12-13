from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..memory import VectorMemory
from ..state import AgentState

class ReportingAgent:
    def __init__(self):
        self.memory = VectorMemory()
        self.llm = ChatOpenAI(model="gpt-5-nano")

    def run(self, state: AgentState):
        """
        Reporting Agent Node logic.
        Synthesizes all gathered data into a final markdown report.
        """
        print("--- Reporting Agent ---")
        ticker = state.get("ticker")
        
        # 1. Retrieve all context from Shared Memory
        print(f"Retrieving context for {ticker}...")
        # detailed query to get both types of docs
        results = self.memory.similarity_search(f"financial analysis and market news for {ticker}", k=10)
        
        context_text = ""
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            context_text += f"-- Source: {source} --\n{doc.page_content}\n\n"
            
        
        mode = state.get("investor_mode", "Neutral")
        print(f"Generating final report in '{mode}' mode...")
        
        system_prompt = f"""You are a professional financial reporter. 
        Your goal is to write a comprehensive financial report for {{ticker}} based on the provided gathered context.
        
        **TONE INSTRUCTION**: The user has requested a **{mode}** perspective. 
        - If 'Bullish': Highlight growth potential, strengths, and upside opportunities.
        - If 'Bearish': Focus on risks, valuation concerns, and competitive threats.
        - If 'Neutral': Maintain a balanced, objective view.
        
        The report must follow this exact Markdown structure:
        
        # Financial Report: {{ticker}}
        
        ## 1. Executive Summary
        (A concise 150-word overview of the company's current status and outlook)
        
        ## 2. Company Snapshot
        (Brief description of sector, market position, and key competitors)
        
        ## 3. Key Financial Indicators
        (Highlight P/E, Market Cap, Price trends, and other metrics found in the context)
        
        ## 4. Recent News & Sentiment
        (Summarize key events and overall market sentiment)
        
        ## 5. Risks & Opportunities
        (Bull case vs Bear case)
        
        ## 6. Final Perspective
        (A concluding statement reflecting the {mode} view)
        
        If some information is missing in the context, state that it was not available.
        Do not make up numbers.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context Data:\n{context}")
        ])
        
        chain = prompt | self.llm
        report = chain.invoke({"ticker": ticker, "context": context_text})
        report_content = report.content
        
        print("Report generation complete.")
        
        # 3. Update State
        return {
            "final_report": report_content,
            "messages": [HumanMessage(content=f"Report generated for {ticker}.")]
        }

def reporter_node(state: AgentState):
    agent = ReportingAgent()
    return agent.run(state)
