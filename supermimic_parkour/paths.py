"""Runtime data locations, independent of the original training repository."""

import os
from pathlib import Path


def default_dataset_dir() -> Path:
    """Resolve user data relative to the invocation directory, not site-packages."""
    return Path(os.environ.get("SUPERMIMIC_DATASET_DIR", "data/php_motion")).expanduser().resolve()
