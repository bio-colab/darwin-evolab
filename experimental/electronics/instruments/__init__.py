"""Cheap benchtop instruments. They read waveforms; they do not simulate circuits."""
from .oscilloscope import measure_transient, measure_waveform

__all__ = ["measure_waveform", "measure_transient"]
