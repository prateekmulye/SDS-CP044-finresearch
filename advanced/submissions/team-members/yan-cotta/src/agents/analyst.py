"""
Analyst Agent - Quantitative financial analysis specialist.

The Analyst agent focuses on retrieving and analyzing financial metrics,
valuation ratios, and fundamental data.
"""

import logging
from typing import Optional

from crewai import Agent

from src.agents.base import BaseAgentFactory, create_llm
from src.config.settings import get_settings
from src.tools.financial_data import FinancialDataTool
from src.tools.memory import MemoryTool


logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Factory and wrapper for the Analyst Agent.
    
    The Analyst agent:
    - Retrieves current stock prices and metrics
    - Analyzes valuation ratios and fundamentals
    - Contextualizes data relative to benchmarks
    - Reports exact figures with proper precision
    """
    
    AGENT_NAME = "analyst"
    
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        financial_tool: Optional[FinancialDataTool] = None
    ):
        """
        Initialize the Analyst agent factory.
        
        Args:
            memory_tool: Optional shared memory tool instance
            financial_tool: Optional financial data tool instance
        """
        self._memory_tool = memory_tool or MemoryTool()
        self._financial_tool = financial_tool or FinancialDataTool()
        self._agent: Optional[Agent] = None
    
    def create(self) -> Agent:
        """
        Create and return the Analyst agent.
        
        Returns:
            Configured Analyst Agent instance
        """
        if self._agent is not None:
            return self._agent
        
        settings = get_settings()
        
        # Analyst uses zero temperature for numerical precision
        llm = create_llm(
            model=settings.worker_model,
            temperature=settings.analyst_temperature
        )
        
        # Analyst gets financial data and memory tools
        tools = [self._financial_tool, self._memory_tool]
        
        self._agent = BaseAgentFactory.create_agent(
            agent_name=self.AGENT_NAME,
            llm=llm,
            tools=tools
        )
        
        logger.info("Analyst agent created successfully")
        return self._agent
    
    @property
    def agent(self) -> Agent:
        """Get the agent instance, creating if necessary."""
        if self._agent is None:
            self.create()
        return self._agent
