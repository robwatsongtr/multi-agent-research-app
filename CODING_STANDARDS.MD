# Coding Standards

## Implementation Standards

- **Separation of concerns**: Maintain clear separation in both files and directories.

- **Clean code**: Keep code clean and readable. Always have one space above a return statement. Keep methods/functions short when possible. Separate function/method calls from return statements - don't combine them on one line.
  - ✅ Good: `response = api.call()\n\nreturn response`
  - ❌ Bad: `return api.call()`

- **Use classes**: Organize functionality with classes to reduce code duplication if possible.

- **Implementation of Features**: Stub out the code and do the boilerplate, but ask if I want to do the logic. If so, let me write the logic, or if not go ahead and implement the feature.

## Documentation Standards

- **Docstrings**: Always include comprehensive docstrings for classes and functions.

- **Inline comments**: Keep inline comments minimal. Only comment tricky logic.
  - ✅ Good: Explaining why something works a certain way
  - ❌ Bad: Explaining what obvious code does

- **Type hints**: Use type hints for all function parameters and return types.
  - Required: Function signatures
  - Optional: Local variables, class variables
  - Run `mypy src/` to verify type correctness (for Python projects)

## MCP-Specific Guidelines

- **No stdout pollution**: Never use `print()` in MCP servers - it corrupts the protocol.
  - ✅ Use: `print(..., file=sys.stderr)` for debugging
  - ✅ Use: `logging` configured to stderr

- **Tool descriptions**: Make tool descriptions clear and specific - the LLM uses these to decide when to call tools.

