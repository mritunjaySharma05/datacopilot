"""Tests for scripts/download_bird.py using small fake archives (no network)."""
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_bird.py"
spec = importlib.util.spec_from_file_location("download_bird", SCRIPT)
bird = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bird)


def make_fake_bird_zip(path: Path) -> bytes:
    """Build a zip shaped like BIRD dev.zip: dev.json + nested dev_databases.zip."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("dev_databases/california_schools/california_schools.sqlite", b"x")
        zf.writestr("dev_databases/financial/financial.sqlite", b"x")
        zf.writestr("__MACOSX/dev_databases/financial/._financial.sqlite", b"x")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("dev_20240627/dev.json", json.dumps([{"question": "a"}, {"question": "b"}]))
        zf.writestr("dev_20240627/dev_databases.zip", inner.getvalue())
    path.write_bytes(outer.getvalue())
    return outer.getvalue()


def fake_response(body: bytes, length: int | None = None) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Length": str(length if length is not None else len(body))}
    response.iter_content.return_value = [body[:10], body[10:]]
    return response


def test_extract_all_unzips_nested_archives(tmp_path):
    zip_path = tmp_path / "dev.zip"
    make_fake_bird_zip(zip_path)
    bird.extract_all(zip_path, tmp_path)
    assert (tmp_path / "dev_20240627/dev_databases/financial/financial.sqlite").exists()
    assert not (tmp_path / "dev_20240627/dev_databases.zip").exists()
    assert zip_path.exists()


def test_find_questions_file_and_count(tmp_path):
    zip_path = tmp_path / "dev.zip"
    make_fake_bird_zip(zip_path)
    bird.extract_all(zip_path, tmp_path)
    questions = bird.find_questions_file(tmp_path)
    assert questions.name == "dev.json"
    assert bird.count_questions(questions) == 2


def test_find_questions_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        bird.find_questions_file(tmp_path)


def test_find_databases_ignores_macosx_and_stray_files(tmp_path):
    zip_path = tmp_path / "dev.zip"
    make_fake_bird_zip(zip_path)
    bird.extract_all(zip_path, tmp_path)
    (tmp_path / "stray.sqlite").write_bytes(b"x")  # not inside a same-named folder
    assert bird.find_databases(tmp_path) == ["california_schools", "financial"]


def test_download_writes_file(tmp_path, monkeypatch):
    body = make_fake_bird_zip(tmp_path / "src.zip")
    monkeypatch.setattr(bird.requests, "get", MagicMock(return_value=fake_response(body)))
    target = bird.download("http://example.test/dev.zip", tmp_path / "out" / "dev.zip")
    assert target.read_bytes() == body
    assert not target.with_suffix(".zip.part").exists()


def test_download_skips_existing_valid_zip(tmp_path, monkeypatch):
    target = tmp_path / "dev.zip"
    make_fake_bird_zip(target)
    get = MagicMock()
    monkeypatch.setattr(bird.requests, "get", get)
    bird.download("http://example.test/dev.zip", target)
    get.assert_not_called()


def test_download_detects_truncation(tmp_path, monkeypatch):
    body = b"0123456789abcdefghij"
    monkeypatch.setattr(
        bird.requests, "get", MagicMock(return_value=fake_response(body, length=999))
    )
    with pytest.raises(IOError, match="Incomplete download"):
        bird.download("http://example.test/dev.zip", tmp_path / "dev.zip")
    assert not (tmp_path / "dev.zip").exists()


def test_main_prints_summary(tmp_path, monkeypatch, capsys):
    body = make_fake_bird_zip(tmp_path / "src.zip")
    monkeypatch.setattr(bird.requests, "get", MagicMock(return_value=fake_response(body)))
    assert bird.main(["--dest", str(tmp_path / "bird")]) == 0
    out = capsys.readouterr().out
    assert "Questions:      2" in out
    assert "Databases:      2" in out
    assert "  - financial" in out
