"""
Cross-platform submission generator for CS336 Assignment 1.
Mirrors all exclusions defined in make_submission.sh.
"""

import os
import zipfile
from pathlib import Path

OUTPUT_FILE = "cs336-spring2025-assignment-1-submission.zip"

EXCLUDED_DIR_PATTERNS = [
    "egg-info",
    "mypy_cache",
    "pytest_cache",
    "build",
    "ipynb_checkpoints",
    "__pycache__",
    ".git",
    ".venv",
    ".vscode",
    "fixtures",
    "_snapshots",
]

EXCLUDED_EXTENSIONS = {
    ".pkl",
    ".pickle",
    ".txt",
    ".log",
    ".json",
    ".out",
    ".err",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".zip",
}


def should_exclude(rel_path: Path) -> bool:
    # Exclude hidden files / dirs
    for part in rel_path.parts:
        if part.startswith(".") and part != ".":
            return True
        for pattern in EXCLUDED_DIR_PATTERNS:
            if pattern in part:
                return True

    # Exclude specific file extensions
    if rel_path.suffix.lower() in EXCLUDED_EXTENSIONS:
        return True

    return False


def create_submission_zip():
    root_dir = Path(".")
    if Path(OUTPUT_FILE).exists():
        Path(OUTPUT_FILE).unlink()

    included_count = 0
    with zipfile.ZipFile(OUTPUT_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in root_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(root_dir)
                if not should_exclude(rel_path):
                    zipf.write(file_path, arcname=str(rel_path).replace("\\", "/"))
                    included_count += 1
                    print(f"  + {rel_path}")

    size_mb = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully compressed {included_count} files into {OUTPUT_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    create_submission_zip()
