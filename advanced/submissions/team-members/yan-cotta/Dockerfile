# =============================================================================
# FinResearch AI - Production Dockerfile
# Multi-Agent Financial Research System
# =============================================================================
#
# Build:
#   docker build -t finresearch-ai:latest .
#
# Run:
#   docker run --rm -it \
#     -e OPENAI_API_KEY=sk-your-key \
#     -v $(pwd)/reports:/app/reports \
#     finresearch-ai:latest AAPL --name "Apple Inc"
#
# Development:
#   docker run --rm -it \
#     -e OPENAI_API_KEY=sk-your-key \
#     -v $(pwd):/app \
#     finresearch-ai:latest --help
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# Install dependencies in a separate stage for smaller final image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VERSION="1.8.4" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY pyproject.toml ./

# Generate requirements and install dependencies
# Using pip for final install to avoid poetry overhead in production
RUN poetry export -f requirements.txt --without-hashes --only main > requirements.txt \
    && pip install --target=/app/deps -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# Minimal image with only runtime dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels for container metadata
LABEL maintainer="Yan Cotta <yan.cotta@example.com>" \
      version="1.0.0" \
      description="FinResearch AI - Multi-Agent Financial Research System" \
      org.opencontainers.image.source="https://github.com/org/finresearch-ai"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application paths
    APP_HOME=/app \
    PYTHONPATH=/app/deps:/app \
    # Disable telemetry
    ANONYMIZED_TELEMETRY=false \
    # ChromaDB settings
    CHROMA_PERSIST_DIR=/app/.chroma_db \
    # Default output directory
    OUTPUT_DIR=/app/reports

# Create non-root user for security
RUN groupadd --gid 1000 finresearch \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home finresearch

# Set working directory
WORKDIR ${APP_HOME}

# Copy dependencies from builder stage
COPY --from=builder /app/deps /app/deps

# Copy application code
COPY --chown=finresearch:finresearch src/ ./src/
COPY --chown=finresearch:finresearch main.py ./

# Create directories for persistence with correct permissions
RUN mkdir -p ${OUTPUT_DIR} ${CHROMA_PERSIST_DIR} \
    && chown -R finresearch:finresearch ${APP_HOME}

# Switch to non-root user
USER finresearch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Volume mounts for persistent data
VOLUME ["/app/reports", "/app/.chroma_db"]

# Entry point - run the CLI
ENTRYPOINT ["python", "main.py"]

# Default command (show help if no args provided)
CMD ["--help"]
