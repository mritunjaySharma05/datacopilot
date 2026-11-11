"""Tests for scripts/download_datasets.py (no network: downloads are faked)."""
import importlib.util
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_datasets.py"
spec = importlib.util.spec_from_file_location("download_datasets", SCRIPT)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return buf.getvalue()


def fake_download_writing(body: bytes):
    """Replacement for download_bird.download that writes `body` to the target."""
    def fake(url, target, force=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return target
    return fake


# --- zip-based datasets ---

def test_get_olist_extracts_all_nine_csvs(tmp_path, monkeypatch):
    body = zip_bytes({name: "a,b\n1,2\n" for name in dd.OLIST_FILES})
    monkeypatch.setattr(dd, "download", fake_download_writing(body))
    dd.get_olist(tmp_path / "olist")
    assert sorted(p.name for p in (tmp_path / "olist").glob("*.csv")) == sorted(dd.OLIST_FILES)


def test_get_olist_reports_missing_files(tmp_path, monkeypatch):
    body = zip_bytes({"olist_orders_dataset.csv": "a\n1\n"})
    monkeypatch.setattr(dd, "download", fake_download_writing(body))
    with pytest.raises(RuntimeError, match="missing expected files"):
        dd.get_olist(tmp_path / "olist")


@pytest.mark.parametrize("code", [401, 403])
def test_get_olist_login_wall_gives_manual_steps(tmp_path, monkeypatch, code):
    def needs_login(url, target, force=False):
        raise requests.HTTPError(response=MagicMock(status_code=code))

    monkeypatch.setattr(dd, "download", needs_login)
    with pytest.raises(dd.ManualStepsRequired, match="kaggle.com/datasets/olistbr"):
        dd.get_olist(tmp_path / "olist")


def test_get_ipl_and_f1_extract(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "download", fake_download_writing(zip_bytes({"1.csv": "x\n", "README.txt": "r"})))
    dd.get_ipl(tmp_path / "ipl")
    dd.get_f1(tmp_path / "f1")
    assert (tmp_path / "ipl" / "1.csv").exists()
    assert (tmp_path / "f1" / "README.txt").exists()


# --- crop (data.gov.in API) ---

def api_page(records, total):
    response = MagicMock()
    response.json.return_value = {"records": records, "total": total}
    return response


def test_fetch_crop_records_pages_until_total(monkeypatch):
    pages = [
        api_page([{"state_name": "Goa", "production": "1"}] * 2, 3),
        api_page([{"state_name": "Kerala", "production": "2"}], 3),
    ]
    get = MagicMock(side_effect=pages)
    monkeypatch.setattr(dd.requests, "get", get)
    df = dd.fetch_crop_records("KEY", page_size=2)
    assert len(df) == 3
    assert [c.kwargs["params"]["offset"] for c in get.call_args_list] == [0, 2]
    assert get.call_args_list[0].kwargs["params"]["api-key"] == "KEY"


def test_fetch_crop_records_raises_api_error(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"error": "Rate limit exceeded"}
    monkeypatch.setattr(dd.requests, "get", MagicMock(return_value=response))
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        dd.fetch_crop_records("KEY")


def test_get_crop_without_key_gives_manual_steps(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    with pytest.raises(dd.ManualStepsRequired, match="DATA_GOV_IN_API_KEY"):
        dd.get_crop(tmp_path / "crop")


def test_get_crop_with_key_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_GOV_IN_API_KEY", "KEY")
    monkeypatch.setattr(dd, "fetch_crop_records", lambda key: pd.DataFrame([{"crop": "Rice"}]))
    dd.get_crop(tmp_path / "crop")
    assert pd.read_csv(tmp_path / "crop" / "crop_production.csv")["crop"].tolist() == ["Rice"]


def test_get_crop_skips_existing_file(tmp_path, monkeypatch):
    target = tmp_path / "crop" / "crop_production.csv"
    target.parent.mkdir()
    target.write_text("crop\nWheat\n")
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    dd.get_crop(tmp_path / "crop")  # no key needed, no error
    assert "Wheat" in target.read_text()


# --- helpers and CLI ---

def test_dir_summary(tmp_path):
    (tmp_path / "a.csv").write_text("12345")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.csv").write_text("123")
    assert dd.dir_summary(tmp_path) == (2, 8)


def test_main_reports_ok_and_manual(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dd, "download", fake_download_writing(zip_bytes({"1.csv": "x\n"})))
    monkeypatch.setattr(dd, "load_dotenv", lambda: None)
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    code = dd.main(["ipl", "crop", "--raw-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1  # crop still needs manual steps
    assert "ipl    ok, " in out
    assert "crop   MANUAL STEPS NEEDED" in out
    assert "data.gov.in" in out


def test_main_rejects_unknown_name(capsys):
    with pytest.raises(SystemExit):
        dd.main(["nope"])
    assert "unknown dataset" in capsys.readouterr().err
