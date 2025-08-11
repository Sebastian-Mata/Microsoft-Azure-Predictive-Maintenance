# Use official Python 3.13 image
FROM python:3.13-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.8.8 /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY data/models/ ./data/models/


# Install dependencies with UV
RUN uv sync --locked

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]