"""
sampling package — class imbalance handling for CoLog

This package provides tools for handling class imbalance in log datasets
using various resampling strategies from the imbalanced-learn library.
"""

from .sampler import ClassImbalanceSolver

__all__ = ['ClassImbalanceSolver']
