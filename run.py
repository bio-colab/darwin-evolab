"""Checkout shim — delegates to the in-repo CLI deterministically.

Always prefers the checkout's src/ over any installed evolab, and purges
stale `evolab` modules from sys.modules so an older installed package can
never shadow the checkout (audit A14 E-05). Prefer the installed
`evolab` command in normal use.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = str(_ROOT / "src")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, _SRC)

_stale = [m for m in sys.modules if m == "evolab" or m.startswith("evolab.")]
for _m in _stale:
    del sys.modules[_m]

from evolab.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
