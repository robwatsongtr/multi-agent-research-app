# Tools vs MCP - Understanding the Difference

## Tool = Lightweight Contract

```
Tool Schema = "I can do X, here's my interface"
Implementation = "Here's how I actually do X"
```

## Two Meanings of "Tool"

### 1. Anthropic API Tool (Claude's perspective)
This is just **metadata/schema** - a JSON object that describes a capability:

```python
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web",
    "input_schema": {...}
}
```

**This is NOT executable code.** It's just information you send to Claude saying "hey, you can ask me to do web_search."

When you pass `tools=[WEB_SEARCH_TOOL]` to the API, Claude can **request** it, but it doesn't actually DO anything by itself.

### 2. Actual Implementation (your code's perspective)
This is the **real Python function** that does the work:

```python
def execute_web_search(query: str, api_key: str) -> list[dict]:
    """This actually calls Tavily API"""
    response = requests.post("https://api.tavily.com/search", ...)
    return response.json()
```

**This is executable code** that makes HTTP requests.

### The Relationship

```
Claude says: "I want to use tool: web_search with query='quantum computing'"
                    ↓
Your code intercepts this request
                    ↓
Your code calls: execute_web_search("quantum computing", api_key)
                    ↓
Your code sends results back to Claude
```

## MCP vs Anthropic Tools

### MCP Server
- Full protocol with handshakes, discovery, etc.
- Runs as separate process
- Handles its own execution
- Can serve multiple tools
- More infrastructure

### Anthropic Tool
- Just a schema definition (JSON)
- Lives in your code
- You handle execution yourself
- Lightweight, no extra process
- Less infrastructure

**Both define a contract/capability**, but MCP is a full client-server architecture, while Anthropic tools are just "hey Claude, if you ask for X, I'll do Y."

## Example in Practice

```python
# The contract
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "input_schema": {...}  # ← What inputs I accept
}

# The implementation
def execute_web_search(query: str) -> dict:
    # ← How I fulfill that contract
    return tavily_api.search(query)
```

You could theoretically build an **MCP server that wraps Tavily**, and it would:
1. Define the same capability
2. Run as a separate process
3. Handle requests via MCP protocol

But for a Python CLI app, just using Anthropic tools directly is simpler and works perfectly fine.

## Summary

- **"tool" in Anthropic sense** = schema/declaration (just JSON metadata)
- **"tool" in general programming sense** = the actual implementation function
- **MCP** = standardized protocol for packaging tools as separate services
- **Direct tool use** = defining schemas and implementing them in your own code

For CLI applications, direct tool use is simpler and sufficient.
