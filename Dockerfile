FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation and copy mode for uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create non-root system user and writable /data directory for optional SQLite state
RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /data && chown -R 10001:10001 /app /data

# Install dependencies first for layer caching
COPY --chown=10001:10001 pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source files
COPY --chown=10001:10001 constants.py main.py mock_data.py schemas.py README.md ./
RUN uv sync --frozen --no-dev

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
