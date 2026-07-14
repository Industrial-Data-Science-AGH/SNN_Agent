# T01 — placeholder image, just enough to prove build -> push -> deploy
# works end-to-end (docs/plans/T01-azure-infra-and-cd-pipeline.md). T02
# replaces main.py and requirements.txt inside this same shape; the
# Dockerfile itself is not expected to change much.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
