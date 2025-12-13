from langgraph.graph import StateGraph, END
from .state import AgentState
from .agents.manager import manager_node
from .agents.researcher import researcher_node
from .agents.analyst import analyst_node
from .agents.reporter import reporter_node

def create_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("manager", manager_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reporter", reporter_node)

    # 2. Add Edges
    # Entry point is Manager
    workflow.set_entry_point("manager")

    # Manager triggers BOTH Researcher and Analyst (Parallel Fan-Out)
    workflow.add_edge("manager", "researcher")
    workflow.add_edge("manager", "analyst")

    # Both Researcher and Analyst go to Reporter (Fan-In/Sync)
    workflow.add_edge("researcher", "reporter")
    workflow.add_edge("analyst", "reporter")

    # Reporter is the final step
    workflow.add_edge("reporter", END)

    # 3. Compile
    app = workflow.compile()
    return app
