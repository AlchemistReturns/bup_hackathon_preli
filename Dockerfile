FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=3.12 \
    PATH="/app/.venv/bin:$PATH"

COPY requirements.txt .
RUN uv venv .venv && uv pip install -r requirements.txt --python .venv

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
