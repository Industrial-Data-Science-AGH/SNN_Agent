"""Placeholder FastAPI app — T01's CD-pipeline proof.

Exists only to give the GitHub Actions workflow
(`.github/workflows/deploy.yml`) and the Terraform-provisioned Container App
(`cloud/infra/container_app.tf`) something real to build, push, and deploy
before any real application code exists. T02
(`docs/plans/T02-fastapi-dashboard-app.md`) replaces this module with the
real ingest/dashboard app inside this same Dockerfile shape — keep this file
minimal, it is not meant to accumulate routes.
"""

from fastapi import FastAPI

app = FastAPI(title="wakeup-ai-cloud-dashboard (placeholder)")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe.

    Confirms the placeholder image deployed and is serving traffic — T01's
    acceptance gate checks this endpoint, not real dashboard behavior.
    """
    return {"status": "ok"}
