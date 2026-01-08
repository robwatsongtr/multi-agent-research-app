# Agent Definition

## What is an Agent?

From a programmatic implementation perspective, **an agent is a class that wraps an LLM API call with:**

1. **A specialized system prompt** - Defines role and behavior
2. **Input processing** - Takes domain-specific input (e.g., a research subtask)
3. **API orchestration** - Calls the LLM with context and constraints
4. **Output parsing** - Extracts and validates structured responses
5. **Error handling** - Manages failures gracefully

## Code Structure

```python
class Agent:
    def __init__(self, client, system_prompt):
        self.client = client
        self.system_prompt = system_prompt

    def execute(self, input_data):
        # Call LLM with specialized prompt
        response = self.client.call(
            system=self.system_prompt,
            user_message=input_data
        )

        # Parse and validate output
        result = self.parse_and_validate(response)

        return result
```

## Key Insight

An agent is **not** the LLM itself. It's the **scaffolding around the LLM** that gives it:

- **A specific purpose** (system prompt)
- **A defined interface** (what goes in, what comes out)
- **Reliability guarantees** (validation, error handling)

It transforms a general-purpose LLM into a specialized component with a contract.

## In This Project

Our agents follow this pattern:

- `BaseAgent` - Provides common API calling and response parsing
- `CoordinatorAgent` - Specializes in breaking queries into subtasks
- `ResearcherAgent` - Specializes in researching individual subtasks
- `SynthesizerAgent` - Specializes in combining findings
- `CriticAgent` - Specializes in validating quality

Each agent has:
- A system prompt (defined in `config/prompts.yaml`)
- A domain-specific method (e.g., `coordinate()`, `research()`)
- JSON schema validation for outputs
- Error handling for edge cases
