#!/usr/bin/env python3
"""
FinResearch AI - Production CLI Entry Point

Command-line interface for running financial research workflows.

Usage:
    python main.py AAPL                      # Research Apple Inc
    python main.py TSLA --name "Tesla Inc"   # Research with custom name
    python main.py MSFT --sequential         # Use sequential process
    python main.py GOOGL --output report.md  # Custom output filename
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import get_settings
from src.crew import FinResearchCrew, SequentialFinResearchCrew
from src.tools.memory import MemoryTool


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def validate_environment() -> bool:
    """
    Validate that required environment variables are set.
    
    Returns:
        True if environment is valid, False otherwise
    """
    settings = get_settings()
    
    if not settings.openai_api_key:
        print("\n" + "=" * 60)
        print("ERROR: OPENAI_API_KEY not configured")
        print("=" * 60)
        print("\nPlease set your OpenAI API key:")
        print("  1. Create a .env file in the project root")
        print("  2. Add: OPENAI_API_KEY=sk-your-key-here")
        print("\nOr set the environment variable:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print("=" * 60 + "\n")
        return False
    
    return True


def print_banner() -> None:
    """Print application banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ███████╗██╗███╗   ██╗██████╗ ███████╗███████╗███████╗ █████╗ ██╗   ║
║   ██╔════╝██║████╗  ██║██╔══██╗██╔════╝██╔════╝██╔════╝██╔══██╗██║   ║
║   █████╗  ██║██╔██╗ ██║██████╔╝█████╗  ███████╗█████╗  ███████║██║   ║
║   ██╔══╝  ██║██║╚██╗██║██╔══██╗██╔══╝  ╚════██║██╔══╝  ██╔══██║██║   ║
║   ██║     ██║██║ ╚████║██║  ██║███████╗███████║███████╗██║  ██║██║   ║
║   ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝   ║
║                                                                      ║
║          Multi-Agent Financial Research System v1.0.0                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="finresearch",
        description="FinResearch AI - Multi-Agent Financial Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL                          Research Apple Inc
  python main.py TSLA --name "Tesla Inc"       Research with company name
  python main.py MSFT --sequential             Use sequential process
  python main.py GOOGL -o google_report.md     Custom output filename
  python main.py NVDA --verbose                Enable verbose logging
        """
    )
    
    # Required arguments
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to research (e.g., AAPL, TSLA, MSFT)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Company name (defaults to ticker symbol)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output filename for the report (default: auto-generated)"
    )
    
    parser.add_argument(
        "-s", "--sequential",
        action="store_true",
        help="Use sequential process instead of hierarchical"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress banner and progress output"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (default: stdout only)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running research"
    )
    
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Clear ChromaDB memory before starting (prevents context pollution)"
    )
    
    return parser


def run_research(args: argparse.Namespace) -> int:
    """
    Execute the research workflow.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger = logging.getLogger(__name__)
    
    ticker = args.ticker.strip().upper()
    company_name = args.name or ticker
    
    if not args.quiet:
        print(f"\nStarting research for: {company_name} ({ticker})")
        print(f"   Process: {'Sequential' if args.sequential else 'Hierarchical'}")
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
    
    try:
        # Create appropriate crew type
        if args.sequential:
            crew = SequentialFinResearchCrew(
                ticker=ticker,
                company_name=company_name,
                verbose=args.verbose
            )
        else:
            crew = FinResearchCrew(
                ticker=ticker,
                company_name=company_name,
                verbose=args.verbose
            )
        
        # Execute research
        result = crew.run()
        
        # Save report
        report_path = crew.save_report(result, filename=args.output)
        
        if not args.quiet:
            print("\n" + "=" * 60)
            print("RESEARCH COMPLETE")
            print("=" * 60)
            print(f"\nReport saved to: {report_path}")
            print("\n--- REPORT PREVIEW ---\n")
            print(result[:2000])
            if len(result) > 2000:
                print("\n... [truncated, see full report] ...")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Research interrupted by user")
        print("\n\nResearch interrupted by user")
        return 1
        
    except Exception as e:
        logger.exception(f"Research failed for {ticker}")
        print(f"\nERROR: {type(e).__name__}: {e}")
        return 1


def reset_memory(quiet: bool = False) -> bool:
    """
    Reset ChromaDB memory to prevent context pollution between sessions.
    
    Args:
        quiet: Suppress output messages
        
    Returns:
        True if reset successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        memory_tool = MemoryTool()
        
        if memory_tool._collection is None:
            if not quiet:
                print("[WARNING] Memory system not available. Nothing to reset.")
            return True
        
        result = memory_tool._clear()
        
        if not quiet:
            print(f"[OK] {result}")
        
        logger.info("Memory reset completed successfully")
        return True
        
    except Exception as e:
        logger.exception("Failed to reset memory")
        if not quiet:
            print(f"[ERROR] Failed to reset memory: {type(e).__name__}: {e}")
        return False


def main() -> int:
    """
    Main entry point.
    
    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        level="DEBUG" if args.verbose else args.log_level,
        log_file=args.log_file
    )
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Handle memory reset
    if args.reset_memory:
        if not args.quiet:
            print("\n" + "-" * 60)
            print("MEMORY RESET")
            print("-" * 60)
        
        if not reset_memory(quiet=args.quiet):
            return 1
        
        if not args.quiet:
            print("-" * 60 + "\n")
    
    # Dry run - just validate
    if args.dry_run:
        print("Configuration valid. Dry run complete.")
        return 0
    
    # Run research
    return run_research(args)


if __name__ == "__main__":
    sys.exit(main())
