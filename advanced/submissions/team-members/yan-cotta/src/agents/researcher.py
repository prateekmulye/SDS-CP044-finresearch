"""
Researcher Agent - Qualitative market research specialist.

The Researcher agent focuses on gathering and synthesizing news,
sentiment, and qualitative information about companies.
"""

import logging
from typing import Optional

from crewai import Agent

from src.agents.base import BaseAgentFactory, create_llm
from src.config.settings import get_settings
from src.tools.news_search import NewsSearchTool
from src.tools.memory import MemoryTool


logger = logging.getLogger(__name__)


class ResearcherAgent:
    """
    Factory and wrapper for the Researcher Agent.
    
    The Researcher agent:
    - Searches for recent news and developments
    - Analyzes market sentiment
    - Identifies key narratives and trends
    - Verifies sources before including them
    """
    
    AGENT_NAME = "researcher"
    
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        news_tool: Optional[NewsSearchTool] = None
    ):
        """
        Initialize the Researcher agent factory.
        
        Args:
            memory_tool: Optional shared memory tool instance
            news_tool: Optional news search tool instance
        """
        self._memory_tool = memory_tool or MemoryTool()
        self._news_tool = news_tool or NewsSearchTool()
        self._agent: Optional[Agent] = None
    
    def create(self) -> Agent:
        """
        Create and return the Researcher agent.
        
        Returns:
            Configured Researcher Agent instance
        """
        if self._agent is not None:
            return self._agent
        
        settings = get_settings()
        
        # Researcher uses higher temperature for creative synthesis
        llm = create_llm(
            model=settings.worker_model,
            temperature=settings.researcher_temperature
        )
        
        # Researcher gets news search and memory tools
        tools = [self._news_tool, self._memory_tool]
        
        self._agent = BaseAgentFactory.create_agent(
            agent_name=self.AGENT_NAME,
            llm=llm,
            tools=tools
        )
        
        logger.info("Researcher agent created successfully")
        return self._agent
    
    @property
    def agent(self) -> Agent:
        """Get the agent instance, creating if necessary."""
        if self._agent is None:
            self.create()
        return self._agent
