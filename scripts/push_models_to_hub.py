"""Starter script for uploading trained models to the Hugging Face Hub."""

from __future__ import annotations

import os


def main() -> None:
    """Validate expected environment variables before implementation."""
    required = ["HF_TOKEN", "HF_REPO_ID"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}")


if __name__ == "__main__":
    main()
