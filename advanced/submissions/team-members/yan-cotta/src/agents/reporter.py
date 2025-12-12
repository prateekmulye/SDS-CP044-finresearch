"""
Reporter Agent - Investment report synthesis specialist.

The Reporter agent combines qualitative and quantitative findings
into a professional, structured investment research report.
"""

import logging
from typing import Optional

from crewai import Agent

from src.agents.base import BaseAgentFactory, create_llm
from src.config.settings import get_settings
from src.tools.memory import MemoryTool


logger = logging.getLogger(__name__)


class ReporterAgent:
    """
    Factory and wrapper for the Reporter Agent.
    
    The Reporter agent:
    - Retrieves findings from team memory
    - Synthesizes research and analysis into coherent narrative
    - Structures reports in professional format
    - Includes appropriate disclaimers and caveats
    """
    
    AGENT_NAME = "reporter"
    
    def __init__(self, memory_tool: Optional[MemoryTool] = None):
        """
        Initialize the Reporter agent factory.
        
        Args:
            memory_tool: Optional shared memory tool instance
        """
        self._memory_tool = memory_tool or MemoryTool()
        self._agent: Optional[Agent] = None
    
    def create(self) -> Agent:
        """
        Create and return the Reporter agent.
        
        Returns:
            Configured Reporter Agent instance
        """
        if self._agent is not None:
            return self._agent
        
        settings = get_settings()
        
        # Reporter uses balanced temperature for structured writing
        llm = create_llm(
            model=settings.worker_model,
            temperature=settings.reporter_temperature
        )
        
        # Reporter only needs memory tool to access team findings
        tools = [self._memory_tool]
        
        self._agent = BaseAgentFactory.create_agent(
            agent_name=self.AGENT_NAME,
            llm=llm,
            tools=tools
        )
        
        logger.info("Reporter agent created successfully")
        return self._agent
    
    @property
    def agent(self) -> Agent:
        """Get the agent instance, creating if necessary."""
        if self._agent is None:
            self.create()
        return self._agent
