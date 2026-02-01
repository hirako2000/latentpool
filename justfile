# Help
default:
    just --list

# Run all checks
check: lint analysis test

# Fix all auto-fixable linting and formatting issues
fix:
    uv run ruff check --fix .

# Check linting and formatting without fixing 
lint:
    uv run ruff check .

# Run static type analysis
analysis:
    uv run pyright

# Run the test suite with coverage
test *args:
    uv run pytest  {{args}} --benchmark-min-rounds=100

# Install/Sync dependencies
setup:
    uv sync