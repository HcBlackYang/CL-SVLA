# MFM/models/__init__.py

"""
Model package initialization file.
Exposes the core model architectures for the MFM project.
"""
from .improved_unified_model import TrueEndToEndDualViewVLA
from .eval_unified_model import EvalTrueEndToEndDualViewVLA

__all__ = [
    "TrueEndToEndDualViewVLA",      # Improved end-to-end model for training
    "EvalTrueEndToEndDualViewVLA"   # Model version adapted for memory-efficient evaluation
]