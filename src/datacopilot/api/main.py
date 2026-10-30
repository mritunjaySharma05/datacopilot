"""FastAPI entry point."""
from fastapi import FastAPI

app = FastAPI(title="DataCopilot")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
