"""
Base agent factory and shared utilities.

Provides common functionality for agent creation and configuration loading.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from crewai import Agent
from langchain_openai import ChatOpenAI

from src.config.settings import get_settings, AGENTS_CONFIG_PATH


logger = logging.getLogger(__name__)


def load_agent_config() -> Dict[str, Any]:
    """
    Load agent configurations from YAML file.
    
    Returns:
        Dictionary with agent configurations
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not AGENTS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Agent config not found: {AGENTS_CONFIG_PATH}")
    
    with open(AGENTS_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded agent config from {AGENTS_CONFIG_PATH}")
    return config


def create_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> ChatOpenAI:
    """
    Create an LLM instance with specified configuration.
    
    Args:
        model: Model name (defaults to settings.worker_model)
        temperature: Temperature setting (defaults to 0.5)
        
    Returns:
        Configured ChatOpenAI instance
    """
    settings = get_settings()
    
    return ChatOpenAI(
        model=model or settings.worker_model,
        temperature=temperature if temperature is not None else 0.5,
        api_key=settings.openai_api_key or None  # Let it use env var if not set
    )


class BaseAgentFactory:
    """
    Factory base class for creating agents.
    
    Provides common configuration loading and agent creation logic.
    """
    
    _config: Optional[Dict[str, Any]] = None
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get cached agent configuration."""
        if cls._config is None:
            cls._config = load_agent_config()
        return cls._config
    
    @classmethod
    def _get_agent_config(cls, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        config = cls.get_config()
        
        if agent_name not in config:
            raise ValueError(f"Agent '{agent_name}' not found in configuration")
        
        return config[agent_name]
    
    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        llm: ChatOpenAI,
        tools: Optional[List[Any]] = None
    ) -> Agent:
        """
        Create an agent from configuration.
        
        Args:
            agent_name: Name of agent in config (e.g., 'researcher')
            llm: Language model instance
            tools: List of tools to assign to agent
            
        Returns:
            Configured Agent instance
        """
        config = cls._get_agent_config(agent_name)
        
        agent = Agent(
            role=config['role'],
            goal=config['goal'].strip(),
            backstory=config['backstory'].strip(),
            llm=llm,
            tools=tools or [],
            allow_delegation=config.get('allow_delegation', False),
            verbose=config.get('verbose', True),
            memory=True,
        )
        
        logger.info(f"Created agent: {config['role']}")
        return agent
