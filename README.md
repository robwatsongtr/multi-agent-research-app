# Multi-Agent Research Assistant

A collaborative AI research system where specialized agents work together to research topics, synthesize findings, and validate claims. Built with **Pydantic models** for type safety and **robust LLM parsing** to handle non-deterministic outputs.

## Features

- 🤖 **Multi-agent collaboration** - Four specialized AI agents (Coordinator, Researcher, Synthesizer, Critic)
- 🔍 **Web search integration** - Automated research using Tavily API
- 📊 **Type-safe data flow** - Pydantic models for validation and IDE autocomplete
- 🛡️ **Robust LLM parsing** - Two-stage parsing handles markdown, code blocks, and format variations
- ✅ **Quality control** - Built-in critic agent validates research quality
- 🧪 **Comprehensive testing** - Unit tests, integration tests, and LLM evals

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

### Data Flow with Pydantic

All inter-agent communication uses validated Pydantic models:

```python
# Coordinator returns
CoordinatorResponse(subtasks=["task1", "task2", "task3"])

# Researcher returns (per subtask)
ResearchResult(
    subtask="...",
    findings=[Finding(claim="...", source="https://...", details="...")]
)

# Synthesizer returns
SynthesizedReport(
    summary="...",
    sections=[SynthesisSection(title="...", content="...", sources=[...])],
    key_insights=["...", "..."]
)

# Critic returns
CriticReview(
    overall_quality="...",
    issues=[CriticIssue(type="...", description="...", severity="...")],
    suggestions=["..."],
    needs_more_research=False
)
```

## Installation

### Prerequisites

- Python 3.12+
- Anthropic API key
- Tavily API key for web search

### Setup

```bash
# Clone and setup
git clone <your-repo-url>
cd multi-agent-research-app
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cat > .env << EOF
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
EOF

# Install pre-commit hooks (optional)
pre-commit install
```

## Quick Start

```bash
# Basic usage
python main.py "What are the latest developments in quantum computing?"

# Verbose mode (detailed logging)
python main.py "What are the latest developments in quantum computing?" --verbose
```

### Agent Implementation

Agents inherit from `BaseAgent` and implement domain logic:

```python
class ResearcherAgent(BaseAgent):
    def research(self, subtask: str, tools, tool_executor) -> ResearchResult:
        # Call Claude API
        response = self.call_claude(subtask, tools=tools, tool_executor=tool_executor)

        # Two-stage parsing
        json_text = extract_json_from_text(self.parse_response(response))
        result_dict = json.loads(json_text)

        # Pydantic validation
        return ResearchResult(**result_dict)
```

System prompts in `config/prompts.yaml` define behavior without code changes.

## Project Structure

```
multi-agent-research-app/
├── agents/
│   ├── base.py            # BaseAgent with API calls & tool handling
│   ├── models.py          # Pydantic models for all data structures
│   ├── parsing.py         # Two-stage LLM parsing (extract + validate)
│   ├── coordinator.py     # Query → subtasks
│   ├── researcher.py      # Subtask → findings (uses web_search tool)
│   ├── synthesizer.py     # Findings → report
│   └── critic.py          # Report → quality review
├── orchestration/
│   └── workflow.py        # Coordinates agent execution
├── config/
│   ├── prompts.yaml       # System prompts for each agent
│   └── settings.py        # API keys, environment config
├── tools/
│   └── web_search.py      # Tavily web search integration
├── tests/
│   ├── test_agents.py     # Unit tests (mocked API)
│   ├── test_researcher.py # Researcher-specific tests
│   ├── test_workflow.py   # Integration tests (mocked API)
│   └── evals/             # LLM evals (real API calls)
│       └── test_workflow_evals.py
└── main.py                # CLI entry point
```

## Extending the System

### Adding a New Agent

1. **Define Pydantic model** in `agents/models.py`:
```python
class FactCheckResult(BaseModel):
    verified: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
```

2. **Create agent** in `agents/fact_checker.py`:
```python
from agents.base import BaseAgent
from agents.models import FactCheckResult
from agents.parsing import extract_json_from_text
import json

class FactCheckerAgent(BaseAgent):
    def verify(self, claim: str, sources: list[str]) -> FactCheckResult:
        response = self.call_claude(f"Claim: {claim}\nSources: {sources}")

        # Two-stage parsing
        json_text = extract_json_from_text(self.parse_response(response))
        result_dict = json.loads(json_text)

        return FactCheckResult(**result_dict)
```

3. **Add prompt** to `config/prompts.yaml`:
```yaml
fact_checker: |
  Verify claims against sources. Return JSON:
  {"verified": true/false, "confidence": 0-1, "reasoning": "..."}
```

4. **Integrate** into `orchestration/workflow.py`

## Testing

```bash
# Fast tests (unit + integration with mocks)
pytest tests/ -v

# All tests including LLM evals (costs tokens)
pytest tests/ -v -m ""

# Type checking
mypy agents/ orchestration/

# Pre-commit hooks (runs on every commit)
pre-commit run --all-files
```

**Test Philosophy:**
- **Unit tests** - Mock API, test parsing logic and Pydantic validation
- **Integration tests** - Mock API, test agent coordination
- **LLM evals** - Real API calls, test output quality properties (not exact matches)

**Pre-commit:** Runs mypy, pytest, and code formatters automatically on commit (see `PRE_COMMIT_SETUP.md`)

## Configuration

**Environment:** `.env` file with API keys
```bash
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

**Agent Behavior:** Edit `config/prompts.yaml`
```yaml
coordinator: |
  Break queries into 3-5 subtasks (changed from 2-4)
  Prioritize recent information from last 6 months
```

**Workflow:** Modify `orchestration/workflow.py` for iteration, multi-round research, etc.

MIT License - see LICENSE file for details

## Contributing

1. Follow coding standards in `CODING_STANDARDS.md`
2. Add tests for new features
3. Update documentation
4. Ensure pre-commit hooks pass

## Acknowledgments

- Built with [Anthropic Claude](https://www.anthropic.com/claude) API
- Web search powered by [Tavily](https://tavily.com/)
