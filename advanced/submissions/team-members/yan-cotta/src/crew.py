"""
FinResearch Crew - Hierarchical multi-agent orchestration.

This module provides the main crew assembly and execution logic,
coordinating the Manager, Researcher, Analyst, and Reporter agents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from crewai import Crew, Task, Process

from src.agents.manager import ManagerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.analyst import AnalystAgent
from src.agents.reporter import ReporterAgent
from src.config.settings import get_settings, TASKS_CONFIG_PATH
from src.tools.memory import MemoryTool
from src.tools.news_search import NewsSearchTool
from src.tools.financial_data import FinancialDataTool


logger = logging.getLogger(__name__)


def load_tasks_config() -> Dict[str, Any]:
    """
    Load task configurations from YAML file.
    
    Returns:
        Dictionary with task configurations
    """
    if not TASKS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Tasks config not found: {TASKS_CONFIG_PATH}")
    
    with open(TASKS_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded tasks config from {TASKS_CONFIG_PATH}")
    return config


class FinResearchCrew:
    """
    Main crew orchestrator for financial research.
    
    Coordinates a hierarchical multi-agent workflow:
    1. Manager receives the research request
    2. Manager delegates to Researcher (qualitative) and Analyst (quantitative)
    3. Workers save findings to shared memory
    4. Reporter synthesizes findings into final report
    
    Attributes:
        ticker: Stock ticker symbol being researched
        company_name: Full company name (optional, for context)
    """
    
    def __init__(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the research crew.
        
        Args:
            ticker: Stock ticker symbol to research
            company_name: Optional company name for context
            verbose: Whether to enable verbose logging
        """
        self.ticker = ticker.strip().upper()
        self.company_name = company_name or self.ticker
        self.verbose = verbose
        
        self._settings = get_settings()
        self._tasks_config = load_tasks_config()
        
        # Shared tools (single instances for all agents)
        self._memory_tool = MemoryTool()
        self._news_tool = NewsSearchTool()
        self._financial_tool = FinancialDataTool()
        
        # Agent factories
        self._manager_factory = ManagerAgent(memory_tool=self._memory_tool)
        self._researcher_factory = ResearcherAgent(
            memory_tool=self._memory_tool,
            news_tool=self._news_tool
        )
        self._analyst_factory = AnalystAgent(
            memory_tool=self._memory_tool,
            financial_tool=self._financial_tool
        )
        self._reporter_factory = ReporterAgent(memory_tool=self._memory_tool)
        
        # Crew instance (created on run)
        self._crew: Optional[Crew] = None
        
        logger.info(f"FinResearchCrew initialized for {self.ticker}")
    
    def _format_task_description(self, template: str) -> str:
        """Format task template with ticker and company name."""
        return template.format(
            ticker=self.ticker,
            company_name=self.company_name
        )
    
    def _create_tasks(self) -> list[Task]:
        """
        Create all tasks for the research workflow.
        
        Returns:
            List of configured Task instances
        """
        # Get agents
        researcher = self._researcher_factory.create()
        analyst = self._analyst_factory.create()
        reporter = self._reporter_factory.create()
        
        # Research Task
        research_config = self._tasks_config['research_task']
        research_task = Task(
            description=self._format_task_description(research_config['description']),
            expected_output=research_config['expected_output'].strip(),
            agent=researcher
        )
        
        # Analysis Task
        analysis_config = self._tasks_config['analysis_task']
        analysis_task = Task(
            description=self._format_task_description(analysis_config['description']),
            expected_output=analysis_config['expected_output'].strip(),
            agent=analyst
        )
        
        # Report Task (depends on research and analysis)
        report_config = self._tasks_config['report_task']
        report_task = Task(
            description=self._format_task_description(report_config['description']),
            expected_output=report_config['expected_output'].strip(),
            agent=reporter,
            context=[research_task, analysis_task]  # Gets output from both
        )
        
        logger.info("Created 3 tasks: research, analysis, report")
        return [research_task, analysis_task, report_task]
    
    def _create_crew(self) -> Crew:
        """
        Assemble the crew with all agents and tasks.
        
        Returns:
            Configured Crew instance
        """
        # Get all agents
        manager = self._manager_factory.create()
        researcher = self._researcher_factory.agent
        analyst = self._analyst_factory.agent
        reporter = self._reporter_factory.agent
        
        # Create tasks
        tasks = self._create_tasks()
        
        # Assemble crew with hierarchical process
        crew = Crew(
            agents=[researcher, analyst, reporter],
            tasks=tasks,
            manager_agent=manager,
            process=Process.hierarchical,
            verbose=self.verbose,
            memory=True,
            planning=True,  # Enable planning for better coordination
        )
        
        logger.info("Crew assembled with hierarchical process")
        return crew
    
    def run(self) -> str:
        """
        Execute the research workflow.
        
        Returns:
            Final research report as string
            
        Raises:
            Exception: If crew execution fails
        """
        logger.info(f"Starting research for {self.ticker} ({self.company_name})")
        
        # Clear previous memory for fresh research
        self._memory_tool._run("clear")
        
        # Create and run the crew
        self._crew = self._create_crew()
        
        try:
            result = self._crew.kickoff()
            
            logger.info(f"Research completed for {self.ticker}")
            return str(result)
            
        except Exception as e:
            logger.exception(f"Crew execution failed for {self.ticker}")
            raise
    
    def save_report(self, content: str, filename: Optional[str] = None) -> Path:
        """
        Save the research report to file.
        
        Args:
            content: Report content to save
            filename: Optional custom filename
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"report_{self.ticker}_{timestamp}.md"
        
        output_path = self._settings.output_path / filename
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Report saved to {output_path}")
        return output_path


class SequentialFinResearchCrew(FinResearchCrew):
    """
    Alternative crew using sequential process instead of hierarchical.
    
    Useful when:
    - Manager delegation isn't needed
    - Simpler, more predictable execution is preferred
    - Debugging individual agent behavior
    """
    
    def _create_crew(self) -> Crew:
        """Create crew with sequential process."""
        researcher = self._researcher_factory.create()
        analyst = self._analyst_factory.create()
        reporter = self._reporter_factory.create()
        
        tasks = self._create_tasks()
        
        crew = Crew(
            agents=[researcher, analyst, reporter],
            tasks=tasks,
            process=Process.sequential,
            verbose=self.verbose,
            memory=True,
        )
        
        logger.info("Crew assembled with sequential process")
        return crew
