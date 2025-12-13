"""
Memory Tool - ChromaDB-based vector memory for agent collaboration.

Provides persistent memory storage allowing agents to save and retrieve
context, enabling knowledge sharing across the research workflow.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from crewai.tools import BaseTool
from pydantic import Field

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

from src.config.settings import get_settings
from src.tools.base import ToolError


logger = logging.getLogger(__name__)


class MemoryTool(BaseTool):
    """
    Tool for storing and retrieving research context using ChromaDB.
    
    Enables agents to:
    - Save findings for team reference
    - Retrieve relevant context from previous research
    - Share information across the research workflow
    """
    
    name: str = "memory_tool"
    description: str = (
        "Store and retrieve research context from team memory. "
        "Use 'save:<category>:<content>' to save information. "
        "Use 'retrieve:<query>' to search for relevant stored information. "
        "Categories: 'news', 'metrics', 'analysis', 'general'. "
        "Example save: 'save:news:Apple announced new iPhone with AI features' "
        "Example retrieve: 'retrieve:Apple recent announcements'"
    )
    
    _client: Optional[Any] = None
    _collection: Optional[Any] = None
    
    def __init__(self, **kwargs):
        """Initialize the memory tool with ChromaDB connection."""
        super().__init__(**kwargs)
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Initialize ChromaDB client and collection."""
        if chromadb is None:
            logger.warning("ChromaDB not installed. Memory features disabled.")
            return
        
        try:
            settings = get_settings()
            
            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=str(settings.chroma_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"description": "FinResearch AI agent memory"}
            )
            
            logger.info(f"ChromaDB initialized at {settings.chroma_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._client = None
            self._collection = None
    
    def _run(self, command: str) -> str:
        """
        Execute a memory operation.
        
        Args:
            command: Memory command in format 'operation:args'
                    - 'save:<category>:<content>'
                    - 'retrieve:<query>'
                    - 'list' - show recent entries
                    - 'clear' - clear all memory
            
        Returns:
            Operation result or error message
        """
        if self._collection is None:
            return "ERROR: Memory system not available. ChromaDB may not be installed."
        
        command = command.strip()
        
        try:
            if command.startswith("save:"):
                return self._save(command[5:])
            elif command.startswith("retrieve:"):
                return self._retrieve(command[9:])
            elif command == "list":
                return self._list_recent()
            elif command == "clear":
                return self._clear()
            else:
                return (
                    "ERROR: Unknown command. Use:\n"
                    "  - save:<category>:<content>\n"
                    "  - retrieve:<query>\n"
                    "  - list\n"
                    "  - clear"
                )
        except Exception as e:
            logger.exception(f"Memory operation failed: {command}")
            return f"ERROR: Memory operation failed: {type(e).__name__}"
    
    def _save(self, content: str) -> str:
        """Save content to memory."""
        parts = content.split(":", 1)
        
        if len(parts) < 2:
            return "ERROR: Save format is 'save:<category>:<content>'"
        
        category = parts[0].strip().lower()
        text = parts[1].strip()
        
        if not text:
            return "ERROR: Cannot save empty content"
        
        valid_categories = ['news', 'metrics', 'analysis', 'general']
        if category not in valid_categories:
            category = 'general'
        
        # Generate unique ID
        doc_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Store in ChromaDB
        self._collection.add(
            documents=[text],
            metadatas=[{
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "source": "agent"
            }],
            ids=[doc_id]
        )
        
        logger.debug(f"Saved to memory: {doc_id}")
        return f"[OK] Saved to memory [{category}]: {text[:100]}{'...' if len(text) > 100 else ''}"
    
    def _retrieve(self, query: str) -> str:
        """Retrieve relevant content from memory."""
        if not query.strip():
            return "ERROR: Empty query provided"
        
        results = self._collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            return f"No relevant memories found for: {query}"
        
        lines = [
            f"MEMORY RETRIEVAL: {query}",
            "=" * 50,
            ""
        ]
        
        for i, (doc, meta) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0]
        ), 1):
            category = meta.get('category', 'unknown')
            timestamp = meta.get('timestamp', 'unknown')
            lines.extend([
                f"[{i}] Category: {category}",
                f"    Saved: {timestamp}",
                f"    Content: {doc}",
                ""
            ])
        
        return "\n".join(lines)
    
    def _list_recent(self, limit: int = 10) -> str:
        """List recent memory entries."""
        # Get all entries (ChromaDB doesn't support easy sorting)
        results = self._collection.get(
            limit=limit,
            include=['documents', 'metadatas']
        )
        
        if not results['documents']:
            return "Memory is empty."
        
        lines = [
            "RECENT MEMORY ENTRIES",
            "=" * 50,
            ""
        ]
        
        for i, (doc, meta) in enumerate(zip(
            results['documents'],
            results['metadatas']
        ), 1):
            category = meta.get('category', 'unknown') if meta else 'unknown'
            lines.append(f"[{i}] [{category}] {doc[:80]}{'...' if len(doc) > 80 else ''}")
        
        return "\n".join(lines)
    
    def _clear(self) -> str:
        """Clear all memory entries."""
        # Delete and recreate collection
        settings = get_settings()
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.create_collection(
            name=settings.chroma_collection_name,
            metadata={"description": "FinResearch AI agent memory"}
        )
        
        logger.info("Memory cleared")
        return "Memory cleared successfully."
    
    # Convenience methods for direct Python access (not via LLM)
    
    def save_context(self, category: str, content: str) -> bool:
        """Direct Python method to save context."""
        result = self._run(f"save:{category}:{content}")
        return result.startswith("[OK]")
    
    def get_context(self, query: str) -> List[str]:
        """Direct Python method to retrieve context."""
        if self._collection is None:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=5
        )
        
        return results['documents'][0] if results['documents'] else []
    
    @classmethod
    def reset_all(cls) -> bool:
        """
        Class method to reset all memory.
        
        Creates a temporary instance and clears the collection.
        Useful for CLI operations and testing.
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            instance = cls()
            if instance._collection is None:
                logger.warning("Memory not available for reset")
                return False
            instance._clear()
            return True
        except Exception as e:
            logger.exception(f"Failed to reset memory: {e}")
            return False
