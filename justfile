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
    uv run pytest {{args}} --benchmark-min-rounds=100

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

# Phase 1 & 2: Hydrate MEV and sample Normals
hydrate:
    uv run latent data hydrate

# Run health report on ingested raw data
hydrate-check:
    uv run latent data check

# Transform raw JSONs to Silver Parquet
prepare:
    uv run latent data prepare

# Join Silver with labels to create Gold set
label:
    uv run latent data label

# Convert Gold Parquet to PyTorch Geometric tensors
process: 
    uv run latent data process

# -- Viz & Diagnostics --

# Visualize a specific transaction flow
viz-tx hash:
    uv run latent viz flow {{hash}}

# High-level health check across all layers (Silver, Gold, Tensors)
viz-health:
    uv run latent viz health-check

# Diagnostic: Graph connectivity and outlier detection
viz-silver-structure:
    uv run latent viz silver structure

# Diagnostic: Node hub distribution and graph diversity
viz-silver-diversity:
    uv run latent viz silver diversity

# Diagnostic: Sequence depth of token swaps
viz-silver-depth:
    uv run latent viz silver depth

# Diagnostic: Feature distribution and value skew
viz-silver-features:
    uv run latent viz silver health

# Diagnostic: Block-level transaction density and bursts
viz-silver-temporal:
    uv run latent viz silver temporal

# -- Utils --

# Wipes all local caches and temporary artifacts (unix only)
clean:
    rm -rf .pytest_cache .ruff_cache .pyright_cache htmlcov .coverage
    find . -type d -name "__pycache__" -exec rm -rf {} +