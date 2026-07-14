"""cloud/app — the single FastAPI container serving the ingest API, the
read API, and the server-rendered dashboard (ADR-0008).

Deployed as the package `app` inside the Container App (see Dockerfile);
imported as `cloud.app` when tests run from the repo root (conftest.py adds
the repo root to sys.path). Every module inside uses relative imports
(`from .auth import ...`) so both import paths resolve identically.
"""
