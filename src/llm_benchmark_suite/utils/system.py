"""System metadata and reproducibility helpers."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Optional


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_info() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
