from typing import TypedDict, Annotated, List, Union
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Shared state for the FinResearch AI graph.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    ticker: str
    research_summary: str
    financial_data: dict
    final_report: str
    next_step: str
