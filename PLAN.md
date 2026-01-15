# PLAN.md - Multi-Agent Research Assistant

## Project Overview
Build a multi-agent research system where specialized AI agents collaborate to research topics, synthesize findings, and validate claims. Each agent is a Python class that calls the Anthropic API with specialized prompts and responsibilities.

## Core Concepts
- **Agent**: Python class that makes API calls to Claude with a specialized system prompt
- **Orchestration**: Main workflow that coordinates agent handoffs and data flow
- **MCP Tools**: Existing tools (web_search, notes) that agents can use during their API calls

## Architecture
```
/agents
  /__init__.py
  /base.py              # Base Agent class with common functionality
  /coordinator.py       # Breaks queries into research subtasks
  /researcher.py        # Executes research using web_search
  /synthesizer.py       # Combines findings into coherent summary
  /critic.py            # Validates claims and identifies gaps
/orchestration
  /__init__.py
  /workflow.py          # Main orchestration logic
/config
  /prompts.yaml         # System prompts for each agent
  /settings.py          # API keys, model configs
/tests
  /test_agents.py       # Unit tests for individual agents
  /test_workflow.py     # Integration tests
/examples
  /simple_research.py   # Example usage
main.py                 # CLI entry point
requirements.txt
README.md
```

## Implementation Phases

### Phase 1: Foundation
**Goal**: Get basic agent infrastructure working with Coordinator

**Tasks**:
1. Set up project structure
2. Create `agents/base.py` with BaseAgent class
   - `__init__` takes anthropic client and system prompt
   - `call_claude()` method wraps API calls
   - `parse_response()` helper for extracting content
3. Create `config/prompts.yaml` with initial Coordinator prompt
4. Implement `agents/coordinator.py`
   - Takes user query
   - Returns JSON list of 2-4 research subtasks
5. Create simple `main.py` CLI to test Coordinator
6. Test: "Research the impact of AI on software development"
   - Should return structured list of subtasks

**Success Criteria**: Can run `python main.py "some query"` and get back a list of subtasks

### Phase 2: Research Agent
**Goal**: Add agent that can execute individual research subtasks

**Tasks**:
1. Implement `agents/researcher.py`
   - Takes a single subtask
   - Uses web_search tool via Anthropic API
   - Returns structured research findings (claims + sources)
2. Add Researcher system prompt to `prompts.yaml`
3. Update `orchestration/workflow.py` to connect Coordinator → Researcher
   - Loop through subtasks
   - Call Researcher for each one
   - Collect all findings
4. Test end-to-end: Query → Subtasks → Research findings

**Success Criteria**: Can research a query and get back findings with sources for each subtask

### Phase 2.5: Progress Logging (Current)
**Goal**: Add real-time progress updates during research execution

**Context**: Research workflows can take several minutes with coordinator and researcher making multiple API calls. Users need visibility into what's happening.

**Tasks**:
1. Configure Python logging module in `main.py`
   - Add logging setup with clean format: `[%(levelname)s] %(message)s`
   - Add `--verbose` CLI flag to toggle between INFO and DEBUG levels
   - Default to INFO level (high-level progress only)
   - Keep existing print statements for final results display
2. Add logging to `orchestration/workflow.py`
   - INFO: "Breaking query into subtasks..."
   - INFO: "Starting research on X subtasks..."
   - INFO: "[X/Y] Researching subtask: {subtask}"
   - INFO: "Completed subtask X/Y"
   - INFO: Tool executor: "Searching web for: {query}"
   - DEBUG: Tool executor: "Found X results"
3. Add logging to agents
   - `coordinator.py`: INFO for "Coordinator analyzing query...", "Generated X subtasks"
   - `researcher.py`: DEBUG for subtask processing details
   - `base.py`: DEBUG for "Calling Claude API...", INFO for "Claude requested tool: {tool_name}"
4. Add logging to `tools/web_search.py`
   - DEBUG: "Executing Tavily search: {query}"
   - ERROR: Search failures
5. Test with and without `--verbose` flag
6. Verify existing tests still pass

**Technical Details**:
- Use Python's built-in `logging` module (stdlib, no installation needed)
- Logging outputs to stderr (best practice - separates diagnostic info from results)
- Two log levels:
  - INFO: Important progress updates user should see
  - DEBUG: Detailed technical information for troubleshooting
- Usage:
  - `python main.py "query"` → Shows progress updates (INFO level)
  - `python main.py "query" --verbose` → Shows detailed tracing (DEBUG level)

**Success Criteria**:
- Users see real-time progress updates during multi-minute research runs
- Progress updates are informative but not overwhelming
- `--verbose` flag provides detailed tracing for debugging
- No disruption to existing functionality or test suite

### Phase 3: Synthesizer Agent
**Goal**: Combine research findings into coherent output

**Tasks**:
1. Implement `agents/synthesizer.py`
   - Takes all research findings
   - Combines into structured summary
   - Organizes by themes/topics
   - Preserves source citations
2. Add Synthesizer system prompt to `prompts.yaml`
3. Update workflow: Coordinator → Researcher → Synthesizer
4. Test: Full pipeline produces readable research report

**Success Criteria**: Get a well-organized research report with proper citations

### Phase 4: Critic Agent
**Goal**: Add validation and quality control

**Tasks**:
1. Implement `agents/critic.py`
   - Reviews synthesized report
   - Identifies unsupported claims
   - Flags contradictions
   - Suggests gaps in research
2. Add Critic system prompt to `prompts.yaml`
3. Update workflow to include Critic review
4. Add optional iteration: if Critic finds gaps, trigger more research
5. Test: Deliberately use vague query to see Critic catch issues

**Success Criteria**: Critic can identify problems and suggest improvements

### Phase 5: Polish & Documentation
**Goal**: Make it production-ready and shareable

**Tasks**:
1. Add comprehensive error handling
2. Implement retry logic for API calls
3. Add logging throughout
4. Write tests for each agent
5. Create example scripts
6. Write detailed README with:
   - Architecture explanation
   - How agents work
   - How to extend with new agents
   - Example outputs
7. Add CLI improvements (progress indicators, better output formatting)

## Key Technical Decisions

### Agent Communication Pattern
- Agents pass structured data (Python dicts/dataclasses)
- Coordinator returns: `list[str]` (subtasks)
- Researcher returns: `list[ResearchFinding]` with sources
- Synthesizer returns: `ResearchReport` with sections
- Critic returns: `CriticReview` with issues/suggestions

### Prompt Engineering Strategy
- Store all system prompts in `prompts.yaml` for easy iteration
- Each prompt includes:
  - Role definition
  - Expected input format
  - Required output format (prefer JSON)
  - Example of good output
- Use YAML anchors for shared prompt sections

### Error Handling
- Wrap all API calls in try/except
- Implement exponential backoff for rate limits
- Validate agent outputs (ensure JSON parseable, required fields present)
- If agent fails, log and either retry or skip gracefully

### Testing Strategy
- Unit tests: Mock Anthropic API, test each agent's logic
- Integration tests: Test workflow with real API (mark as slow)
- Fixtures: Save example API responses for consistent testing

## Extension Ideas (Post-MVP)
- Add memory: Store past research in Notes app via MCP
- Add fact-checking agent that verifies specific claims
- Add source quality agent that ranks source reliability
- Add visualization agent that creates diagrams/charts
- Make it interactive: User can provide feedback and trigger re-research
- Add streaming: Show progress as agents work

## Success Metrics
1. **Functional**: Can research a topic and produce a coherent report
2. **Quality**: Reports include proper citations and catch obvious errors
3. **Architecture**: Easy to add new agents with clear interfaces
4. **Code Quality**: Well-tested, documented, follows standards
5. **Shareable**: README is good enough to post publicly

## First Command to Run
```bash
# After Claude Code scaffolds the structure
python main.py "What are the latest developments in quantum computing?"
```

This should trigger the full pipeline and produce a research report.

## Notes for Claude Code
- Start with Phase 1 only - don't build everything at once
- Focus on getting one agent working end-to-end before adding complexity
- Use your existing web_search MCP tool - agents should leverage it
- Keep prompts in YAML so we can iterate without code changes
- Add type hints throughout for clarity
- Follow CODING_STANDARDS.md for all implementation
