"""
Application settings and configuration management.

This module provides centralized configuration using Pydantic for validation
and environment variable loading.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden via environment variables.
    """
    
    # API Keys
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM access"
    )
    
    # Model Configuration
    manager_model: str = Field(
        default="gpt-4o-mini",
        description="Model for the Manager agent (requires strong reasoning)"
    )
    worker_model: str = Field(
        default="gpt-3.5-turbo",
        description="Model for worker agents (Researcher, Analyst, Reporter)"
    )
    
    # Temperature Settings
    manager_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for Manager (low for consistent delegation)"
    )
    researcher_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for Researcher (higher for creative synthesis)"
    )
    analyst_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for Analyst (zero for numerical precision)"
    )
    reporter_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Temperature for Reporter (balanced for structured writing)"
    )
    
    # Memory / ChromaDB Configuration
    chroma_persist_dir: str = Field(
        default=".chroma_db",
        description="Directory for ChromaDB persistence"
    )
    chroma_collection_name: str = Field(
        default="finresearch_memory",
        description="ChromaDB collection name"
    )
    
    # Output Configuration
    output_dir: str = Field(
        default="./reports",
        description="Directory for generated reports"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for stdout only)"
    )
    
    # Request Configuration
    max_news_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum news articles to retrieve"
    )
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for external API requests in seconds"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "FINRESEARCH_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def chroma_path(self) -> Path:
        """Get the ChromaDB persistence path as a Path object."""
        return Path(self.chroma_persist_dir)
    
    @property
    def output_path(self) -> Path:
        """Get the output directory as a Path object."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings instance (cached for performance)
    """
    return Settings()


# Paths relative to this module
CONFIG_DIR = Path(__file__).parent
AGENTS_CONFIG_PATH = CONFIG_DIR / "agents.yaml"
TASKS_CONFIG_PATH = CONFIG_DIR / "tasks.yaml"
