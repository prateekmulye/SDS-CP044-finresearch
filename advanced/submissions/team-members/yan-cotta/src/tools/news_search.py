"""
News Search Tool - Wrapper for DuckDuckGo News Search.

Provides news discovery with source verification and structured output
for qualitative research.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import Field

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from src.tools.base import ToolError


logger = logging.getLogger(__name__)


class NewsSearchTool(BaseTool):
    """
    Tool for searching recent news articles about companies or topics.
    
    Provides access to:
    - Recent news headlines and summaries
    - Source verification and credibility indicators
    - Publication dates and source attribution
    """
    
    name: str = "news_search_tool"
    description: str = (
        "Searches for recent news articles about a company or financial topic. "
        "Returns headlines, summaries, sources, and publication dates. "
        "Input: company name, ticker, or topic (e.g., 'Apple earnings', 'TSLA news'). "
        "Use this for NEWS, SENTIMENT, and QUALITATIVE information, not numbers."
    )
    
    max_results: int = Field(default=10, description="Maximum articles to return")
    
    # Known credible financial news sources
    CREDIBLE_SOURCES: List[str] = [
        "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
        "marketwatch.com", "seekingalpha.com", "fool.com", "barrons.com",
        "yahoo.com", "finance.yahoo.com", "businessinsider.com", "forbes.com",
        "thestreet.com", "investopedia.com", "benzinga.com"
    ]
    
    def _run(self, query: str) -> str:
        """
        Search for news articles about the specified topic.
        
        Args:
            query: Search query (company name, topic, etc.)
            
        Returns:
            Formatted string with news articles or error message
        """
        if DDGS is None:
            logger.error("duckduckgo-search library not installed")
            return "ERROR: duckduckgo-search library not installed. Run: pip install duckduckgo-search"
        
        query = self._normalize_query(query)
        
        try:
            articles = self._fetch_news(query)
            return self._format_output(query, articles)
        except ToolError as e:
            logger.warning(f"Tool error for query '{query}': {e.message}")
            return f"ERROR: {e.message}"
        except Exception as e:
            logger.exception(f"Unexpected error searching news for '{query}'")
            return f"ERROR: Unexpected error searching news: {type(e).__name__}"
    
    def _normalize_query(self, query: str) -> str:
        """Normalize and enhance the search query."""
        if not query or not query.strip():
            raise ToolError(self.name, "Empty search query provided")
        
        query = query.strip()
        
        # Add financial context if not present
        financial_terms = ['stock', 'share', 'market', 'earnings', 'investor', 'financial', 'trading']
        if not any(term in query.lower() for term in financial_terms):
            query = f"{query} stock market news"
        
        return query
    
    def _fetch_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch news articles from DuckDuckGo.
        
        Args:
            query: Enhanced search query
            
        Returns:
            List of article dictionaries
            
        Raises:
            ToolError: If search fails
        """
        try:
            articles = []
            
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    keywords=query,
                    max_results=self.max_results + 5  # Fetch extra for filtering
                ))
            
            if not results:
                logger.info(f"No news results for query: {query}")
                return []
            
            for result in results[:self.max_results]:
                article = self._process_article(result)
                if article:
                    articles.append(article)
            
            return articles
            
        except Exception as e:
            raise ToolError(
                self.name,
                f"Failed to search news: {str(e)}",
                original_error=e
            )
    
    def _process_article(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process and validate a single article."""
        url = raw.get('url', '')
        
        if not url:
            return None
        
        source = raw.get('source', self._extract_source(url))
        
        return {
            "title": raw.get('title', 'No title'),
            "url": url,
            "source": source,
            "date": raw.get('date', 'Unknown'),
            "snippet": raw.get('body', 'No summary available.'),
            "is_verified": self._verify_source(url),
        }
    
    def _extract_source(self, url: str) -> str:
        """Extract source domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return "Unknown"
    
    def _verify_source(self, url: str) -> bool:
        """Check if source is from a known credible financial news outlet."""
        try:
            domain = self._extract_source(url)
            return any(credible in domain for credible in self.CREDIBLE_SOURCES)
        except Exception:
            return False
    
    def _format_output(self, query: str, articles: List[Dict[str, Any]]) -> str:
        """Format articles into structured output for LLM consumption."""
        if not articles:
            return f"No recent news found for: {query}"
        
        verified_count = sum(1 for a in articles if a['is_verified'])
        
        lines = [
            f"NEWS SEARCH RESULTS: {query}",
            "=" * 60,
            f"Found {len(articles)} articles ({verified_count} from verified sources)",
            "",
        ]
        
        for i, article in enumerate(articles, 1):
            verified_badge = "[VERIFIED]" if article['is_verified'] else "[UNVERIFIED]"
            
            lines.extend([
                f"[{i}] {article['title']}",
                f"    Source: {article['source']} ({verified_badge})",
                f"    Date: {article['date']}",
                f"    URL: {article['url']}",
                f"    Summary: {article['snippet'][:200]}{'...' if len(article['snippet']) > 200 else ''}",
                "",
            ])
        
        lines.extend([
            "-" * 60,
            "NOTE: Articles marked [VERIFIED] are from known financial news sources.",
            "      Articles marked [UNVERIFIED] should be cross-referenced.",
            f"Search completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ])
        
        return "\n".join(lines)
