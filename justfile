# Help
default:
    just --list

# -- LINT & TESTS --
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

# Open the visual coverage report
coverage:
    open htmlcov/index.html

# --- DATA OPS ---

# Pull "Bronze" traces from Hugging Face
data-pull:
    uv run dvc pull

# Push new traces/models to Hugging Face
data-push:
    uv run dvc push

# Wipe all local caches and temporary artifacts (unix only)
clean:
    rm -rf .pytest_cache .ruff_cache .pyright_cache htmlcov .coverage
    find . -type d -name "__pycache__" -exec rm -rf {} +