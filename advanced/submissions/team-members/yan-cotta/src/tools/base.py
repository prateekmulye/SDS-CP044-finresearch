"""
Base tool class and shared utilities for FinResearch tools.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Standardized result format for all tools."""
    
    success: bool = Field(description="Whether the operation succeeded")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if not self.success:
            return f"ERROR: {self.error}"
        return str(self.data)


class ToolError(Exception):
    """Custom exception for tool errors."""
    
    def __init__(self, tool_name: str, message: str, original_error: Optional[Exception] = None):
        self.tool_name = tool_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{tool_name}] {message}")
