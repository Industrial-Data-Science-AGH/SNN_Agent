from fastapi import FastAPI

app = FastAPI(title="IDS Backend")


@app.get("/health")
def health():
    return {"status": "ok"}
