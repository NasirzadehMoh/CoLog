"""
Dataclasses for groundtruth extraction.

This module contains dataclass definitions used throughout the groundtruth
extraction process.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DatasetAttributes:
    """Represents a dataset attributes (train/valid/test) with keys, messages, sequences, and labels.
    
    Attributes
    ----------
    keys : List[str]
        Ordered list of message IDs in this split.
    messages : Dict[str, Any]
        Dictionary mapping message IDs to message data (tokenized or raw).
    sequences : Dict[str, List[str]]
        Dictionary mapping message IDs to their context sequences.
    labels : Dict[str, int]
        Dictionary mapping message IDs to their labels.
    """
    keys: List[str] = field(default_factory=list)
    messages: Dict[str, Any] = field(default_factory=dict)
    sequences: Dict[str, List[str]] = field(default_factory=dict)
    labels: Dict[str, int] = field(default_factory=dict)
