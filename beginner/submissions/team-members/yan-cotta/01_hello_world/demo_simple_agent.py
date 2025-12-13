#!/usr/bin/env python3
"""
================================================================================
FinResearch AI - Beginner Track Demo: "Hello World" Agent
================================================================================

PURPOSE:
    This script introduces the THREE ATOMIC UNITS of the CrewAI framework:
    1. Agent  - The "who" (a persona with a role, goal, and backstory)
    2. Task   - The "what" (a specific assignment with expected output)
    3. Crew   - The "orchestrator" (manages agents and executes tasks)

LEARNING OBJECTIVES:
    - Understand how to instantiate a single agent
    - Learn what makes a good role, goal, and backstory
    - See how tasks are assigned and executed
    - Observe verbose output to understand agent "reasoning"

BEFORE RUNNING:
    1. Ensure you have installed dependencies: pip install -r requirements.txt
    2. Set your OpenAI API key: export OPENAI_API_KEY="sk-..."
    
Author: Yan Cotta | FinResearch AI Project
================================================================================
"""

import os
import sys
from pathlib import Path

# =============================================================================
# STEP 0A: LOAD ENVIRONMENT VARIABLES FROM .env FILE
# =============================================================================
# WHY WE USE DOTENV: Storing API keys in code is a security anti-pattern.
# The .env file keeps secrets out of version control (it's in .gitignore).
# python-dotenv loads these variables into os.environ automatically.

from dotenv import load_dotenv

# Find the .env file in the project root (navigate up from this script's location)
# This approach works regardless of where you run the script from.
project_root = Path(__file__).resolve().parents[4]  # Go up 4 levels to reach project root
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    # Try current working directory as fallback
    load_dotenv()

# =============================================================================
# STEP 0B: ENVIRONMENT VALIDATION (Safety First!)
# =============================================================================
# WHY THIS MATTERS: The most common error beginners face is forgetting to set
# their API key. This check provides a friendly error instead of a cryptic
# exception from the OpenAI library later in execution.

def validate_environment():
    """
    Validates that required environment variables are set.
    Returns True if valid, exits with helpful message if not.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n" + "=" * 70)
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nTo fix this, run one of the following commands:\n")
        print("  Linux/Mac:  export OPENAI_API_KEY='sk-your-key-here'")
        print("  Windows:    set OPENAI_API_KEY=sk-your-key-here")
        print("\nOr add it to a .env file in the project root.")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    # SECURITY TIP: We check the prefix to catch common copy-paste errors
    # where users might paste a placeholder or malformed key.
    if not api_key.startswith("sk-"):
        print("\n⚠️  WARNING: Your OPENAI_API_KEY doesn't start with 'sk-'.")
        print("   This may indicate an invalid key format.\n")
    
    return True

# Run validation before any imports that might use the API key
validate_environment()

# =============================================================================
# STEP 1: IMPORTS
# =============================================================================
# CONCEPT: We import from 'crewai' for the orchestration framework, and from
# 'langchain_openai' for the LLM wrapper. CrewAI is LLM-agnostic, meaning you
# could swap ChatOpenAI for Anthropic, Ollama, or other providers.

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# =============================================================================
# STEP 2: DEFINE AN AGENT
# =============================================================================
# CONCEPT: An Agent is like hiring a team member. You define:
#   - role: Their job title (affects how they approach problems)
#   - goal: What they're trying to achieve (their motivation)
#   - backstory: Their personality and expertise (shapes their "voice")
#
# BEST PRACTICE: Write backstories in second person ("You are...") as this
# has been shown to improve LLM role-playing performance.

analyst = Agent(
    role='Junior Financial Analyst',
    goal='Summarize financial concepts simply for beginners',
    backstory=(
        "You are an enthusiastic junior analyst who loves teaching. "
        "You have a gift for using simple metaphors to explain complex "
        "stock market terms. You remember what it was like to be confused "
        "by jargon, so you always put clarity first."
    ),
    
    # VERBOSE MODE: Set to True during development to see the agent's
    # internal reasoning process. This is invaluable for debugging!
    # In production, you'd typically set this to False.
    verbose=True,
    
    # DELEGATION: We set allow_delegation=False because this simple agent
    # should execute tasks itself rather than trying to assign them to
    # other agents. For a single-agent crew, delegation would fail anyway.
    allow_delegation=False,
    
    # LLM CONFIGURATION: We use gpt-3.5-turbo for cost efficiency in demos.
    # Temperature 0.7 allows for creative explanations while maintaining
    # coherence. For factual tasks, you'd use temperature=0.
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
)

# =============================================================================
# STEP 3: DEFINE A TASK
# =============================================================================
# CONCEPT: A Task is a specific assignment. The key fields are:
#   - description: What needs to be done (be specific and clear!)
#   - expected_output: What format/structure should the result have
#   - agent: Who is responsible for this task
#
# BEST PRACTICE: The more specific your expected_output, the better the
# results. Vague instructions like "write something good" produce vague output.

explain_task = Task(
    description=(
        "Explain what 'Short Selling' is to a 10-year-old. "
        "Use a lemonade stand analogy to make it relatable. "
        "Avoid any financial jargon or complex terminology."
    ),
    
    # EXPECTED OUTPUT: This acts as a "contract" - the agent will try to
    # match this format. Being specific here dramatically improves results.
    expected_output=(
        "A 3-paragraph explanation using a lemonade stand metaphor. "
        "Paragraph 1: Set up the analogy. "
        "Paragraph 2: Explain the concept. "
        "Paragraph 3: Explain the risk."
    ),
    
    # AGENT ASSIGNMENT: Each task must be assigned to exactly one agent.
    # The agent's role and backstory will influence how they approach this.
    agent=analyst
)

# =============================================================================
# STEP 4: DEFINE THE CREW
# =============================================================================
# CONCEPT: The Crew is the "manager" that orchestrates everything. It:
#   - Holds references to all agents and tasks
#   - Determines execution order via the 'process' parameter
#   - Manages the workflow and collects final output
#
# PROCESS TYPES:
#   - Process.sequential: Tasks run one after another (simplest)
#   - Process.hierarchical: A "manager" agent delegates to others (advanced)

my_crew = Crew(
    agents=[analyst],
    tasks=[explain_task],
    
    # SEQUENTIAL PROCESS: For a single task, this is straightforward.
    # With multiple tasks, they would execute in the order listed.
    process=Process.sequential,
    
    # CREW VERBOSE: Shows high-level execution flow. Combined with agent
    # verbose=True, you get complete visibility into the system.
    verbose=True
)

# =============================================================================
# STEP 5: EXECUTE THE CREW
# =============================================================================
# CONCEPT: kickoff() is the main entry point. It:
#   1. Initializes all agents
#   2. Executes tasks according to the process type
#   3. Returns the final output from the last task
#
# NOTE: This is a synchronous call. For async execution, use kickoff_async().

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 STARTING CREW EXECUTION")
    print("=" * 70)
    print("Watch the verbose output below to see the agent's reasoning...\n")
    
    try:
        result = my_crew.kickoff()
        
        print("\n" + "=" * 70)
        print("✅ FINAL RESULT")
        print("=" * 70)
        print(result)
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ EXECUTION ERROR: {type(e).__name__}")
        print("=" * 70)
        print(f"Details: {e}")
        print("\nCommon fixes:")
        print("  1. Check your OPENAI_API_KEY is valid and has credits")
        print("  2. Ensure you have internet connectivity")
        print("  3. Verify crewai is installed: pip install crewai")
        print("=" * 70 + "\n")
        sys.exit(1)