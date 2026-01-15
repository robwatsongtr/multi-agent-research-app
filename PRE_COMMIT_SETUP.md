# Pre-Commit Setup

This project uses [pre-commit](https://pre-commit.com/) to automatically run checks before each commit.

## What Runs on Commit

Every time you commit, the following checks run automatically:

### 1. **mypy (Type Checking)**
- Verifies all type hints are correct
- Checks your source directories
- Fails if type errors are found

### 2. **pytest (Test Suite)**
- Runs all tests automatically
- Fails if any test fails
- Uses arm64 architecture for compatibility on Apple Silicon

### 3. **Code Quality Checks**
- Trims trailing whitespace
- Fixes end-of-file formatting
- Validates YAML syntax
- Prevents large files from being committed
- Checks for merge conflict markers

## Installation

The hooks are automatically installed when you set up the project:

```bash
# Already done during setup
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

## Usage

### Automatic (on commit)
```bash
git add .
git commit -m "Your message"

# Pre-commit runs automatically:
# ✓ mypy (type checking)
# ✓ pytest (test suite)
# ✓ trim trailing whitespace
# ✓ fix end of files
# ✓ check yaml syntax
# ✓ check for large files
# ✓ check for merge conflicts

# If all checks pass → commit succeeds ✅
# If any check fails → commit is blocked, fix and try again ❌
```

### Manual (run before commit)
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

## Bypassing Hooks (Emergency Only)

**Not recommended**, but if you need to commit without running hooks:

```bash
git commit --no-verify -m "Your message"
```

## What to Do When Hooks Fail

### mypy fails
```
mypy (type checking).....Failed
config/settings.py:30: error: Returning Any from function declared to return "dict[str, str]"
```

**Fix**: Correct the type error in the file, then commit again.

### Trailing whitespace
```
trim trailing whitespace.....Failed
- files were modified by this hook
Fixing agents/base.py
```

**Fix**: The hook auto-fixed it! Just add the file and commit again:
```bash
git add agents/base.py
git commit -m "Your message"
```

## Configuration Files

- `.pre-commit-config.yaml` - Pre-commit hook configuration
- `mypy.ini` - Type checking configuration

## Notes

- Pre-commit creates isolated environments for hooks (separate from your venv)
- First run takes a few minutes to set up environments (cached after that)
- pytest uses `arch -arm64` to ensure compatibility on Apple Silicon
- All tests run on every commit

## Updating Hooks

```bash
# Update to latest hook versions
pre-commit autoupdate

# Re-install hooks after config changes
pre-commit install
```
