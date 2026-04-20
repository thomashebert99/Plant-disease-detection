"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_monitoring_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep prediction monitoring logs out of the repository during tests."""

    monkeypatch.setenv("MONITORING_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(tmp_path / "feedback.jsonl"))
