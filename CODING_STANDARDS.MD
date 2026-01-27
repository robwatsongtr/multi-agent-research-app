# Coding Standards

## Python Project Setup

All Python projects must include:

- **pytest**: Testing framework for all unit and integration tests
  - Run tests with: `pytest tests/ -v`
  - Configure in `pytest.ini` or `pyproject.toml`

- **mypy**: Static type checker for type safety
  - Run type checking with: `mypy src/` (or relevant source directories)
  - Configure in `mypy.ini` with strict settings
  - Use `arch -arm64` prefix on Apple Silicon Macs

- **pydantic**: Data validation and structured data modeling
  - Required for all structured data (API responses, configuration, data transfer objects)
  - Use `BaseModel` for all data structures instead of `dict[str, Any]`
  - Provides automatic validation, type safety, and IDE autocomplete
  - Version: `pydantic>=2.0.0`

- **pre-commit**: Git hooks for automated quality checks
  - Install hooks with: `pre-commit install`
  - Configure in `.pre-commit-config.yaml`
  - Must include: mypy, pytest, trailing whitespace, end-of-file fixer
  - Use local hooks with `arch -arm64` for Python tools on Apple Silicon
  - See @PRE_COMMIT_SETUP.MD for detailed setup and usage instructions

## Implementation Standards

- **Separation of concerns**: Maintain clear separation in both files and directories.

- **Clean code**: Keep code clean and readable. Always have one space above a return statement. Keep methods/functions short when possible. Separate function/method calls from return statements - don't combine them on one line.
  - ✅ Good: `response = api.call()\n\nreturn response`
  - ❌ Bad: `return api.call()`

- **Use classes**: Organize functionality with classes to reduce code duplication if possible.

- **Structured data**: Use Pydantic models for all structured data instead of nested dictionaries.
  - ✅ Good: `class User(BaseModel): name: str; email: str`
  - ❌ Bad: `user: dict[str, Any] = {"name": "...", "email": "..."}`
  - Benefits: Type safety, validation, IDE autocomplete, easier refactoring
  - Access: `user.name` instead of `user['name']`

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
