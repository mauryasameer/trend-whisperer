from unittest.mock import patch

import pytest

from scripts import fetch_data


def test_fetch_fails_cleanly_without_credentials(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fetch_data, "KAGGLE_CREDENTIALS", tmp_path / "nope.json")
    with pytest.raises(SystemExit) as exc_info:
        fetch_data.fetch_rossmann_data()
    assert exc_info.value.code == 1
    assert "Kaggle API credentials not found" in capsys.readouterr().err


def test_fetch_downloads_and_extracts(monkeypatch, tmp_path):
    creds = tmp_path / "kaggle.json"
    creds.write_text("{}")
    monkeypatch.setattr(fetch_data, "KAGGLE_CREDENTIALS", creds)
    monkeypatch.setattr(fetch_data, "DATA_DIR", tmp_path / "data")

    def _fake_run(cmd, check):
        zip_path = (tmp_path / "data") / "rossmann-store-sales.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("train.csv", "Store,Sales\n1,100\n")

    with patch("subprocess.run", side_effect=_fake_run):
        result_dir = fetch_data.fetch_rossmann_data()

    assert (result_dir / "train.csv").is_file()
    assert not (result_dir / "rossmann-store-sales.zip").exists()
