"""rod_eval — reusable accuracy + speed evaluation helpers for the ROD-Dataset.

Adapted from the `test and evaluation/` suite of a separate reference project
(Real-Time-Obstacle-Detector-ComputerVision) so the same metrics can run
headless/locally as well as inside the Kaggle notebooks.
"""
from .accuracy_metrics import calculate_accuracy_metrics
from .time_inference import calculate_time_inference
from .print_and_save_results import print_and_save_results

__all__ = [
    "calculate_accuracy_metrics",
    "calculate_time_inference",
    "print_and_save_results",
]
