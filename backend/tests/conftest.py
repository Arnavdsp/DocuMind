from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

os.environ["MODEL_BACKEND"] = "mock"  # never download real models in tests
os.environ["TRANSLATION_PROVIDER"] = "none"

from app.config import get_settings  # noqa: E402
from app.dependencies import reset_singletons  # noqa: E402


@pytest.fixture()
def temp_data_dir(monkeypatch):
    tmp_dir = tempfile.mkdtemp(prefix="dis_test_")
    monkeypatch.setenv("DATA_DIR", tmp_dir)
    reset_singletons()
    settings = get_settings()
    settings.ensure_dirs()
    yield Path(tmp_dir)
    reset_singletons()
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def client(temp_data_dir):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def read_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()
