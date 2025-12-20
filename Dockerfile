# build-env
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --group inference --no-install-project

# build-app
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src

COPY model/onnx ./model/onnx

COPY main.py ./main.py

EXPOSE 8000

ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["src.app.handler"]
