# T02 — real ingest/dashboard app image, replacing T01's placeholder
# (docs/plans/T02-fastapi-dashboard-app.md, Step 7). Same overall shape T01
# established (install deps, copy app code, run uvicorn); T01's CD pipeline
# (.github/workflows/deploy.yml) and Terraform (cloud/infra/) needed no
# changes to build/deploy this.
#
# App code is copied into /app/app/ (not /app/ directly) because every
# module under cloud/app uses relative imports (`from . import storage`) —
# they need a real package (`app/`, with __init__.py) to resolve against,
# the same way pytest imports this code as `cloud.app.*` from the repo root.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
