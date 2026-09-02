"""Core import must not load side representations."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_import_does_not_load_side_modules():
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import evolab; "
        "loaded = [n for n in sys.modules if n.startswith('evolab.')]; "
        "bad = [n for n in loaded if n in ("
        "'evolab.cgp_logic','evolab.quantum','evolab.llm_mutator',"
        "'evolab.assembly_genome','evolab.cst_genome'"
        ")]; "
        "raise SystemExit(0 if not bad else 'loaded ' + ','.join(bad))"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_lazy_side_export_still_resolves():
    from evolab import EventBus

    assert EventBus is not None
