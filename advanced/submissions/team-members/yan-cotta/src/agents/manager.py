"""
Manager Agent - Orchestrates the research workflow.

The Manager agent is responsible for delegating tasks to specialized
agents and ensuring the research process flows correctly.
"""

import logging
from typing import Optional

from crewai import Agent

from src.agents.base import BaseAgentFactory, create_llm
from src.config.settings import get_settings
from src.tools.memory import MemoryTool


logger = logging.getLogger(__name__)


class ManagerAgent:
    """
    Factory and wrapper for the Manager Agent.
    
    The Manager agent:
    - Coordinates the research workflow
    - Delegates tasks to Researcher, Analyst, and Reporter
    - Ensures quality and completeness of final output
    """
    
    AGENT_NAME = "manager"
    
    def __init__(self, memory_tool: Optional[MemoryTool] = None):
        """
        Initialize the Manager agent factory.
        
        Args:
            memory_tool: Optional shared memory tool instance
        """
        self._memory_tool = memory_tool or MemoryTool()
        self._agent: Optional[Agent] = None
    
    def create(self) -> Agent:
        """
        Create and return the Manager agent.
        
        Returns:
            Configured Manager Agent instance
        """
        if self._agent is not None:
            return self._agent
        
        settings = get_settings()
        
        # Manager uses a more capable model for complex reasoning
        llm = create_llm(
            model=settings.manager_model,
            temperature=settings.manager_temperature
        )
        
        # Manager only needs memory tool to coordinate
        tools = [self._memory_tool]
        
        self._agent = BaseAgentFactory.create_agent(
            agent_name=self.AGENT_NAME,
            llm=llm,
            tools=tools
        )
        
        logger.info("Manager agent created successfully")
        return self._agent
    
    @property
    def agent(self) -> Agent:
        """Get the agent instance, creating if necessary."""
        if self._agent is None:
            self.create()
        return self._agent
