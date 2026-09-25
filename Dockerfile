# Darwin-Evolab: Multi-Domain Evolutionary Optimization & Synthesis Engine
FROM python:3.11-slim AS base

WORKDIR /app

# Install minimal build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition and source tree
COPY pyproject.toml ./
COPY src/ ./src/
COPY examples/ ./examples/

# Install evolab kernel in editable mode with full dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -e ".[full]"

# Verify CLI entrypoint
RUN evolab --version

# Set entrypoint to evolab console script
ENTRYPOINT ["evolab"]
CMD ["--help"]
