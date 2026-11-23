"""Static checks on the Docker setup: pinned versions agree, no secrets or data in images, non-root."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "docker" / "Dockerfile").read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()
DOCKERIGNORE = (ROOT / ".dockerignore").read_text().split()
# Image package -> development package with the same code (CPU-only XGBoost build in the image).
SAME_AS = {"xgboost-cpu": "xgboost"}


def pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, version = line.split("==")
            out[name.strip().lower()] = version.strip()
    return out


@pytest.mark.parametrize("image", ["api", "ui"])
def test_image_requirements_match_dev_pins(image):
    dev = pins(ROOT / "requirements.txt")
    for name, version in pins(ROOT / "docker" / f"requirements-{image}.txt").items():
        assert dev.get(SAME_AS.get(name, name)) == version, name


def test_api_image_has_everything_src_imports():
    api = set(pins(ROOT / "docker" / "requirements-api.txt"))
    for module, package in {"pandas": "pandas", "duckdb": "duckdb", "sqlglot": "sqlglot", "sklearn": "scikit-learn",
                            "xgboost": "xgboost-cpu", "fastapi": "fastapi", "dotenv": "python-dotenv",
                            "requests": "requests"}.items():
        assert package in api, module


def test_build_context_excludes_secrets_and_data():
    for entry in (".env", "data", ".venv", ".git"):
        assert entry in DOCKERIGNORE


def test_dockerfile_copies_no_env_or_data_and_runs_as_non_root():
    copies = [line for line in DOCKERFILE.splitlines() if line.startswith("COPY")]
    assert copies and not any(".env" in c or " data" in c for c in copies)
    targets = re.split(r"^FROM ", DOCKERFILE, flags=re.M)
    for target in ("api", "ui"):
        body = next(t for t in targets if t.split("\n")[0].strip().endswith(f"AS {target}"))
        assert "USER app" in body, target
    assert "ENV GROQ_API_KEY" not in DOCKERFILE and "ARG GROQ_API_KEY" not in DOCKERFILE


def test_compose_passes_only_needed_settings_and_mounts_data_read_only():
    assert "env_file" not in COMPOSE  # would copy every key in .env into the container
    assert "GROQ_API_KEY: ${GROQ_API_KEY:-}" in COMPOSE
    assert "DATA_GOV_IN_API_KEY" not in COMPOSE
    assert "./data/processed:/data:ro" in COMPOSE
    assert "OLLAMA_HOST: http://host.docker.internal:11434" in COMPOSE
