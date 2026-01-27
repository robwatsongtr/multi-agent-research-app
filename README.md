# Multi-Agent Research Assistant

A collaborative AI research system where specialized agents work together to research topics, synthesize findings, and validate claims. Each agent is a Python class that calls the Anthropic Claude API with specialized prompts and responsibilities.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Extending the System](#extending-the-system)
- [Testing](#testing)
- [Configuration](#configuration)
- [Example Output](#example-output)

## Features

- 🤖 **Multi-agent collaboration**: Four specialized AI agents work together to produce comprehensive research
- 🔍 **Web search integration**: Automated web research using Tavily API
- 📊 **Structured output**: Well-organized reports with citations and key insights
- ✅ **Quality control**: Built-in critic agent validates research quality
- 🧪 **Comprehensive testing**: Unit tests, integration tests, and LLM evals
- 🛠️ **Easily extensible**: Clean architecture for adding new agents

## Architecture

The system uses four specialized agents that collaborate in sequence:

```
User Query
    ↓
┌─────────────────┐
│  COORDINATOR    │  Breaks query into 2-4 research subtasks
└────────┬────────┘
         ↓
┌─────────────────┐
│  RESEARCHER     │  Executes each subtask using web search
│  (per subtask)  │  Returns findings with sources
└────────┬────────┘
         ↓
┌─────────────────┐
│  SYNTHESIZER    │  Combines findings into coherent report
│                 │  Organizes by themes, preserves citations
└────────┬────────┘
         ↓
┌─────────────────┐
│   CRITIC        │  Reviews quality, identifies gaps
│                 │  Suggests improvements
└─────────────────┘
```

### Agent Responsibilities

1. **Coordinator Agent**: Analyzes user queries and breaks them into 2-4 focused research subtasks
2. **Researcher Agent**: Executes web searches for each subtask and extracts structured findings with sources
3. **Synthesizer Agent**: Combines all findings into an organized report with sections and key insights
4. **Critic Agent**: Reviews the synthesized report for quality, unsupported claims, and research gaps

## Installation

### Prerequisites

- Python 3.12+
- Anthropic API key (Tier 2 recommended for better rate limits)
- Tavily API key for web search

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd multi-agent-research-app
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
EOF
```

5. Install pre-commit hooks (optional but recommended):
```bash
pre-commit install
```

## Quick Start

### Using the CLI

```bash
# Basic usage
python3 main.py "What are the latest developments in quantum computing?"

# Verbose mode (shows detailed logging)
python3 main.py "What are the latest developments in quantum computing?" --verbose
```

### Using the Python API

```python
from anthropic import Anthropic
from config.settings import load_prompts, get_api_key, get_tavily_api_key
from orchestration.workflow import run_research_workflow

# Initialize
api_key = get_api_key()
tavily_api_key = get_tavily_api_key()
prompts = load_prompts()
client = Anthropic(api_key=api_key)

# Run research
result = run_research_workflow(
    query="What are the environmental impacts of cryptocurrency mining?",
    client=client,
    coordinator_prompt=prompts['coordinator'],
    researcher_prompt=prompts['researcher'],
    synthesizer_prompt=prompts['synthesizer'],
    critic_prompt=prompts['critic'],
    tavily_api_key=tavily_api_key
)

# Access results
print(result['synthesis']['summary'])
for section in result['synthesis']['sections']:
    print(f"{section['title']}: {section['content']}")
```

See `examples/simple_research.py` for a complete example.

## How It Works

### Agent Implementation

Each agent is a Python class that inherits from `BaseAgent`:

```python
class BaseAgent:
    def __init__(self, client: Anthropic, system_prompt: str):
        self.client = client
        self.system_prompt = system_prompt

    def call_claude(self, user_message: str, tools=None, tool_executor=None):
        # Makes API call to Claude with system prompt
        # Handles tool use if tools are provided
        # Returns parsed response
```

Specialized agents (Coordinator, Researcher, Synthesizer, Critic) inherit from `BaseAgent` and add domain-specific logic.

### System Prompts

All agent behavior is defined by system prompts stored in `config/prompts.yaml`. This makes it easy to iterate on agent behavior without changing code.

Example prompt structure:
```yaml
coordinator: |
  You are a Research Coordinator agent...

  Your task: Break user queries into 2-4 focused research subtasks.

  Output format: JSON array of strings
  ["subtask 1", "subtask 2", ...]
```

### Tool Integration

The Researcher agent has access to a `web_search` tool implemented via the Tavily API:

```python
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information...",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        }
    }
}
```

When the Researcher calls this tool, the workflow automatically executes the search and returns results.

### Data Flow

Agents communicate using structured Python dictionaries:

```python
# Coordinator returns
["subtask 1", "subtask 2", "subtask 3"]

# Researcher returns (per subtask)
{
    "subtask": "subtask 1",
    "findings": [
        {
            "claim": "Key finding...",
            "source": "https://example.com",
            "details": "Additional context..."
        }
    ]
}

# Synthesizer returns
{
    "summary": "Overall summary...",
    "sections": [
        {
            "title": "Section Title",
            "content": "Section content...",
            "sources": ["https://source1.com", "https://source2.com"]
        }
    ],
    "key_insights": ["Insight 1", "Insight 2"]
}

# Critic returns
{
    "overall_quality": "Assessment...",
    "issues": [
        {
            "severity": "medium",
            "type": "unsupported_claim",
            "description": "Description...",
            "location": "Section 2"
        }
    ],
    "suggestions": ["Suggestion 1", "Suggestion 2"],
    "needs_more_research": false
}
```

## Project Structure

```
multi-agent-research-app/
├── agents/                 # Agent implementations
│   ├── base.py            # Base agent class
│   ├── coordinator.py     # Query decomposition
│   ├── researcher.py      # Web research execution
│   ├── synthesizer.py     # Report synthesis
│   ├── critic.py          # Quality validation
│   └── json_utils.py      # JSON parsing utilities
├── orchestration/         # Workflow coordination
│   └── workflow.py        # Main orchestration logic
├── config/                # Configuration
│   ├── prompts.yaml       # Agent system prompts
│   └── settings.py        # API keys, settings
├── tools/                 # MCP tools
│   └── web_search.py      # Tavily integration
├── tests/                 # Test suite
│   ├── test_agents.py     # Unit tests
│   ├── test_workflow.py   # Integration tests
│   └── evals/             # LLM quality evals
│       └── test_workflow_evals.py
├── examples/              # Example scripts
│   └── simple_research.py
├── main.py               # CLI entry point
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Extending the System

### Adding a New Agent

1. Create a new agent class in `agents/`:

```python
from agents.base import BaseAgent
from anthropic import Anthropic

class FactCheckerAgent(BaseAgent):
    """Agent that verifies specific claims."""

    def verify_claim(self, claim: str, sources: list[str]) -> dict:
        """Verify a claim against provided sources."""
        user_message = f"Claim: {claim}\nSources: {sources}"
        response = self.call_claude(user_message)

        # Parse and return structured result
        return self.parse_response(response)
```

2. Add system prompt to `config/prompts.yaml`:

```yaml
fact_checker: |
  You are a Fact Checker agent. Your role is to verify claims
  against provided sources and assess their accuracy.

  Return JSON: {"verified": true/false, "confidence": 0-1, "reasoning": "..."}
```

3. Integrate into workflow in `orchestration/workflow.py`:

```python
def run_research_workflow(..., fact_checker_prompt: str):
    # ... existing workflow ...

    # Add fact checking
    fact_checker = FactCheckerAgent(client, fact_checker_prompt)
    for finding in research_results:
        verification = fact_checker.verify_claim(
            finding['claim'],
            finding['sources']
        )
        finding['verified'] = verification
```

### Adding a New Tool

1. Create tool definition in `tools/`:

```python
CUSTOM_TOOL = {
    "name": "custom_tool",
    "description": "Tool description for LLM",
    "input_schema": {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter desc"}
        },
        "required": ["param"]
    }
}

def execute_custom_tool(param: str) -> dict:
    """Execute the custom tool."""
    # Implementation
    return {"result": "..."}
```

2. Pass tool to agent in workflow:

```python
def tool_executor(tool_name: str, tool_input: dict) -> dict:
    if tool_name == "custom_tool":
        return execute_custom_tool(**tool_input)
    # ... other tools ...

agent.call_claude(
    message,
    tools=[CUSTOM_TOOL],
    tool_executor=tool_executor
)
```

## Testing

### Running Tests

```bash
# Run fast tests only (unit + integration with mocks)
pytest tests/ -v

# Run all tests including slow LLM evals (requires API calls)
pytest tests/ -v -m ""

# Run only LLM evals
pytest tests/evals/ -v -m slow

# Run with coverage
pytest tests/ --cov=agents --cov=orchestration
```

### Test Categories

1. **Unit Tests** (`tests/test_agents.py`): Test individual agent logic with mocked API calls
   - Fast, free, test code paths
   - Example: Does coordinator parse JSON correctly?

2. **Integration Tests** (`tests/test_workflow.py`): Test workflow with mocked API responses
   - Fast, free, test component integration
   - Example: Does full workflow complete without errors?

3. **LLM Evals** (`tests/evals/test_workflow_evals.py`): Test real LLM output quality
   - Slow, costs tokens, test real behavior
   - Marked with `@pytest.mark.slow`
   - Example: Do research findings include valid citations?

### Evaluation Pattern

LLM evals test properties, not exact matches:

```python
@pytest.mark.slow
def test_citation_preservation(api_client, tavily_key, prompts):
    """Eval: Citations should flow from research to synthesis."""
    result = run_research_workflow(...)

    # Assert properties, not exact values
    assert len(result['synthesis']['sources']) > 0
    assert all(s.startswith('http') for s in sources)

    # Check quality metrics
    coverage = calculate_citation_coverage(result)
    assert coverage >= 0.8  # 80% threshold
```

### Pre-commit Hooks

The project uses pre-commit hooks for quality assurance:

```bash
# Automatically runs on every commit:
# - mypy (type checking)
# - pytest (test suite)
# - trailing whitespace removal
# - YAML syntax validation
# - end-of-file fixing

# Run manually
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

## Configuration

### Environment Variables

Set in `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-...      # Required
TAVILY_API_KEY=tvly-...           # Required
MODEL=claude-sonnet-4-5-20250929  # Optional, defaults to latest
```

### Adjusting Agent Behavior

Edit system prompts in `config/prompts.yaml`:

```yaml
coordinator: |
  # Modify coordinator behavior
  You are a Research Coordinator...

  # Change subtask count
  Break queries into 3-5 focused subtasks (not 2-4)

  # Add constraints
  Prioritize recent information from the last 6 months
```

### Workflow Configuration

Modify `orchestration/workflow.py` to change agent interaction:

```python
# Add iteration based on critic feedback
if result['critique']['needs_more_research']:
    # Run additional research
    additional_subtasks = generate_followup_tasks(result)
    # ...
```

## Example Output

Input:
```bash
python3 main.py "What are the benefits of Python for data science?"
```

Output:
```
================================================================================
RESEARCH SUBTASKS
================================================================================

1. Core Python libraries and frameworks for data science (NumPy, Pandas,
   Scikit-learn)
2. Python's advantages in data visualization (Matplotlib, Seaborn, Plotly)
3. Integration with machine learning and AI ecosystems

================================================================================
RESEARCH FINDINGS
================================================================================

[Subtask 1]: Core Python libraries and frameworks for data science
Findings: 3

  1. NumPy provides efficient array operations fundamental to data manipulation
     Source: https://numpy.org/doc/stable/user/whatisnumpy.html

  2. Pandas offers powerful DataFrame structures for tabular data analysis
     Source: https://pandas.pydata.org/docs/getting_started/overview.html

  3. Scikit-learn provides comprehensive machine learning algorithms
     Source: https://scikit-learn.org/stable/

...

================================================================================
SYNTHESIZED RESEARCH REPORT
================================================================================

Python has become the dominant language for data science due to its extensive
library ecosystem, intuitive syntax, and strong community support...

────────────────────────────────────────────────────────────────────────────────
Core Libraries and Frameworks
────────────────────────────────────────────────────────────────────────────────

NumPy, Pandas, and Scikit-learn form the foundation of Python's data science
stack. NumPy provides efficient numerical computations...

Sources:
  • https://numpy.org/doc/stable/user/whatisnumpy.html
  • https://pandas.pydata.org/docs/getting_started/overview.html

────────────────────────────────────────────────────────────────────────────────
KEY INSIGHTS
────────────────────────────────────────────────────────────────────────────────

1. Python's extensive library ecosystem reduces development time significantly
2. Integration with Jupyter notebooks enables interactive data exploration
3. Strong community support provides abundant learning resources

================================================================================
CRITIC REVIEW
================================================================================

Overall Quality: Comprehensive overview with well-cited claims

────────────────────────────────────────────────────────────────────────────────
SUGGESTIONS FOR IMPROVEMENT
────────────────────────────────────────────────────────────────────────────────

1. Consider adding performance comparisons with other languages (R, Julia)
2. Include information about Python's limitations for large-scale data

────────────────────────────────────────────────────────────────────────────────
✓ Critic assessment: Research is comprehensive.
────────────────────────────────────────────────────────────────────────────────

================================================================================
Research complete: 3 subtasks, 3 sections
================================================================================
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:

1. Follow the coding standards in `CODING_STANDARDS.md`
2. Add tests for new features
3. Update documentation
4. Ensure pre-commit hooks pass

## Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/claude) API
- Web search powered by [Tavily](https://tavily.com/)
- Inspired by multi-agent AI research patterns
