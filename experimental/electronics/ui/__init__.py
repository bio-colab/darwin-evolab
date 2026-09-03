"""
ui package — Interactive UI/UX Workbench and Virtual Instruments for the darwin-evolab electronics track.
"""
from .workbench_generator import generate_workbench_html, save_workbench_html

__all__ = [
    "generate_workbench_html",
    "save_workbench_html",
]
