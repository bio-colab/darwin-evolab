"""export_dataset.py — Packages Darwin-Evolab into a clean, lightweight Kaggle Dataset archive.

Creates: kaggle_bundle/darwin_evolab_dataset.zip
Excludes: .git, __pycache__, virtualenvs, temporary logs, and heavy binary artifacts.
Includes: src/evolab, reports/, selected scripts/, and empty self_runs.sqlite schema.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path


def create_empty_self_runs_sqlite(db_path: Path) -> None:
    """Initializes empty self_runs.sqlite database with the canonical Phase 1 schema."""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    schema = """
    CREATE TABLE IF NOT EXISTS experiences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        eval_index INTEGER NOT NULL,
        problem_fingerprint TEXT NOT NULL,
        func_name TEXT NOT NULL DEFAULT '',
        target_file TEXT NOT NULL DEFAULT '',
        genome_class TEXT NOT NULL DEFAULT '',
        edit_kinds TEXT NOT NULL DEFAULT '[]',
        edit_loci TEXT NOT NULL DEFAULT '[]',
        n_edits INTEGER NOT NULL DEFAULT 0,
        score REAL NOT NULL,
        fitness_delta REAL,
        is_new_best INTEGER NOT NULL DEFAULT 0,
        passed_holdout INTEGER,
        eval_ms REAL NOT NULL DEFAULT 0.0,
        outcome TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_exp_fp ON experiences(problem_fingerprint);
    CREATE INDEX IF NOT EXISTS idx_exp_fp_outcome ON experiences(problem_fingerprint, outcome);

    CREATE TABLE IF NOT EXISTS self_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        strategy TEXT NOT NULL DEFAULT '',
        seed INTEGER,
        generations INTEGER NOT NULL DEFAULT 0,
        population_size INTEGER NOT NULL DEFAULT 0,
        best_fitness REAL NOT NULL DEFAULT 0.0,
        mean_fitness REAL,
        evals_total INTEGER NOT NULL DEFAULT 0,
        early_stopped INTEGER,
        stagnation_gens INTEGER NOT NULL DEFAULT 0,
        diversity_final REAL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_self_runs_strategy ON self_runs(strategy);

    CREATE TABLE IF NOT EXISTS self_capabilities (
        domain TEXT PRIMARY KEY,
        trials INTEGER NOT NULL DEFAULT 0,
        passes INTEGER NOT NULL DEFAULT 0,
        pass_rate REAL NOT NULL DEFAULT 0.0,
        ci_low REAL NOT NULL DEFAULT 0.0,
        ci_high REAL NOT NULL DEFAULT 1.0,
        updated_at TEXT NOT NULL
    );
    """
    cursor.executescript(schema)
    conn.commit()
    conn.close()
    print(f"[OK] Initialized empty canonical self_runs.sqlite at {db_path}")


def package_kaggle_dataset() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    bundle_dir = repo_root / "kaggle_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    db_path = bundle_dir / "self_runs.sqlite"
    create_empty_self_runs_sqlite(db_path)

    zip_path = bundle_dir / "darwin_evolab_dataset.zip"
    if zip_path.exists():
        zip_path.unlink()

    included_dirs = ["src", "reports", "scripts", "tests"]
    excluded_extensions = {".pyc", ".pyo", ".log", ".tmp"}
    excluded_dir_names = {"__pycache__", ".git", ".idea", ".vscode", "venv", ".venv"}

    total_files = 0
    total_bytes = 0

    print(f"[INFO] Packaging Darwin-Evolab into {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add the empty self_runs.sqlite
        zf.write(db_path, arcname="self_runs.sqlite")
        total_files += 1

        for d_name in included_dirs:
            d_path = repo_root / d_name
            if not d_path.exists():
                continue
            for root, dirs, files in os.walk(d_path):
                dirs[:] = [d for d in dirs if d not in excluded_dir_names]
                for f in files:
                    file_p = Path(root) / f
                    if file_p.suffix in excluded_extensions or file_p.name.startswith("."):
                        continue
                    arc_name = file_p.relative_to(repo_root).as_posix()
                    zf.write(file_p, arcname=arc_name)
                    total_files += 1
                    total_bytes += file_p.stat().st_size

        # Also include top-level configs and docs if present
        for doc_name in ["README.md", "README_ar.md", "pyproject.toml", "setup.py"]:
            doc_p = repo_root / doc_name
            if doc_p.exists():
                zf.write(doc_p, arcname=doc_name)
                total_files += 1
                total_bytes += doc_p.stat().st_size

    mb_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Packaged {total_files} files ({mb_size:.2f} MB) into {zip_path}")
    print(f"          Ready for upload to Kaggle as a Dataset named 'darwin-evolab-telemetry'.")
    return zip_path


if __name__ == "__main__":
    package_kaggle_dataset()
