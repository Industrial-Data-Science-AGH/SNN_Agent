# Backend image: the API and the worker are this same image with a different command.
#   docker build -f rpi_agents/cloud/infra/Dockerfile -t snn-backend .
#   docker run --rm snn-backend api          # or: worker | both
# The build context is the repository root; Dockerfile.dockerignore keeps it to the few files the backend needs.

FROM python:3.12-slim AS deps
WORKDIR /build
COPY requirements-backend.lock .
RUN python -m venv /venv \
 && /venv/bin/pip install --no-cache-dir --require-hashes -r requirements-backend.lock

FROM python:3.12-slim
# Unprivileged user with no home and no shell.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin snn
COPY --from=deps /venv /venv
WORKDIR /app
COPY contracts/ contracts/
COPY rpi_agents/__init__.py rpi_agents/__init__.py
COPY rpi_agents/cloud/ rpi_agents/cloud/
COPY rpi_agents/runtime/ rpi_agents/runtime/
# The vision client reuses the small HTTP transport of the edge package; nothing else of the edge is shipped.
COPY rpi_agents/agent/__init__.py rpi_agents/agent/api.py rpi_agents/agent/
ENV PATH=/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]
ENTRYPOINT ["python", "-m", "rpi_agents.cloud.app.server"]
CMD ["api"]
