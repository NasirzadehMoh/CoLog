"""
extraction package — Ground truth extraction module for CoLog

This package contains the GroundTruth class and related extraction functionality
for building and persisting ground truth datasets.

The main entry point is the GroundTruth facade class, which automatically
dispatches to the appropriate extractor based on dataset type.

For advanced usage, the following base and specialized extractors are also available:
- GroundTruthExtractor: Base class for all extractors
- Type1GroundTruthExtractor: For hadoop, zookeeper datasets
- Type2GroundTruthExtractor: For spark, windows datasets
- Type3GroundTruthExtractor: For bgl datasets
- Type4GroundTruthExtractor: For NER-parsed datasets
- GroundTruthAggregator: For combining multiple datasets
"""

from .main import GroundTruth
from .groundtruth_base import GroundTruthExtractor
from .groundtruth_type1 import Type1GroundTruthExtractor
from .groundtruth_type2 import Type2GroundTruthExtractor
from .groundtruth_type3 import Type3GroundTruthExtractor
from .groundtruth_type4 import Type4GroundTruthExtractor
from .groundtruth_aggregator import GroundTruthAggregator

__all__ = [
    'GroundTruth',
    'GroundTruthExtractor',
    'Type1GroundTruthExtractor',
    'Type2GroundTruthExtractor',
    'Type3GroundTruthExtractor',
    'Type4GroundTruthExtractor',
    'GroundTruthAggregator',
]
