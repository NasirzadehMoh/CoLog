"""
extraction/main.py — GroundTruth extraction class for CoLog

This module provides the main GroundTruth facade class that dispatches to
the appropriate extraction implementation based on dataset type.

The extraction logic has been refactored into separate modules:
    - groundtruth_base.py: Base class with shared functionality
    - groundtruth_type1.py: hadoop, zookeeper (WARN-based labels)
    - groundtruth_type2.py: spark, windows (wordlist-based labeling)
    - groundtruth_type3.py: bgl (Label column where '-' is normal)
    - groundtruth_type4.py: casper-rw, dfrws*, honeynet* (NER parser)
    - groundtruth_aggregator.py: Combines multiple datasets
"""

from utils import ISettingsProvider
from utils import constants
from .groundtruth_type1 import Type1GroundTruthExtractor
from .groundtruth_type2 import Type2GroundTruthExtractor
from .groundtruth_type3 import Type3GroundTruthExtractor
from .groundtruth_type4 import Type4GroundTruthExtractor
from .groundtruth_aggregator import GroundTruthAggregator
import logging


logger = logging.getLogger(__name__)


class GroundTruth(object):
    """
    Build and persist ground truth for a dataset.

    The `GroundTruth` object is a facade that dispatches to the appropriate
    extraction strategy based on dataset type. This maintains backward
    compatibility while the implementation has been refactored into
    separate, specialized modules.

    Dataset method dispatch for ground truth generation:
      - Type1: hadoop, zookeeper (WARN-based labels)
      - Type2: spark, windows (wordlist-based labeling)
      - Type3: bgl (Label column where '-' is normal)
      - Type4: casper-rw, dfrws*, honeynet* (NER parser)
      - Aggregator: Combines multiple datasets

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'hadoop').
    sequence_type : str
        'background' or 'context' to control the sequence building direction.
    window_size : int
        Number of messages to include on each side of the current message.
    train_ratio : float
        Fraction of normal/anomaly records to place in the training set.
    valid_ratio : float
        Fraction of normal/anomaly records to place in validation set.
    sampling_method : str
        Sampling method name.
    datasets_dir : str, optional
        Root directory containing datasets. Default is 'datasets/'.
    force : bool, optional
        If True, forces re-extraction even if files exist. Default is False.
    groundbreaking : bool, optional
        If True, enables multi-label groundbreaking mode. Default is False.
    random_seed : int, optional
        Random seed for reproducibility. Default is 100.
    dry_run : bool, optional
        Preview operations without execution if True. Default is False.
    settings_provider : ISettingsProvider, optional
        Settings provider for dependency injection. Default is None.
    """
    
    def __init__(self, dataset: str, sequence_type: str, window_size: int, train_ratio: float, 
                 valid_ratio: float, sampling_method: str, datasets_dir: str = 'datasets/', 
                 force: bool = False, groundbreaking: bool = False, 
                 random_seed: int = constants.DEFAULT_RANDOM_SEED, dry_run: bool = False, 
                 settings_provider: ISettingsProvider = None) -> None:
        """
        Initialize GroundTruth facade and create appropriate extractor.
        
        This constructor determines the dataset type and instantiates the
        appropriate specialized extractor class.
        """
        self.dataset = dataset
        self.sequence_type = sequence_type
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.sampling_method = sampling_method
        self.datasets_dir = datasets_dir
        self.force = force
        self.groundbreaking = groundbreaking
        self.random_seed = random_seed
        self.dry_run = dry_run
        self.settings_provider = settings_provider
        
        # Strategy mapping for dataset-specific ground truth extraction
        self._strategy_map = {
            'type1': Type1GroundTruthExtractor,
            'type2': Type2GroundTruthExtractor,
            'type3': Type3GroundTruthExtractor,
            'type4': Type4GroundTruthExtractor,
            'all': GroundTruthAggregator
        }
        
        # Create the appropriate extractor instance
        dataset_type = self._get_dataset_type()
        if dataset_type and dataset_type in self._strategy_map:
            extractor_class = self._strategy_map[dataset_type]
            self._extractor = extractor_class(
                dataset=dataset,
                sequence_type=sequence_type,
                window_size=window_size,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                sampling_method=sampling_method,
                datasets_dir=datasets_dir,
                force=force,
                groundbreaking=groundbreaking,
                random_seed=random_seed,
                dry_run=dry_run,
                settings_provider=settings_provider
            )
        else:
            self._extractor = None

    def _get_dataset_type(self) -> str:
        """
        Determine the dataset type based on the dataset name.

        Returns
        -------
        str or None
            Dataset type identifier ('type1', 'type2', 'type3', 'type4', or 'all').
            Returns None if dataset is not recognized.
        """
        if self.dataset in constants.LOGS_TYPE1:
            return 'type1'
        elif self.dataset in constants.LOGS_TYPE2:
            return 'type2'
        elif self.dataset in constants.LOGS_TYPE3:
            return 'type3'
        elif self.dataset in constants.LOGS_TYPE4:
            return 'type4'
        elif self.dataset == 'all':
            return 'all'
        return None

    def extract_groundtruth(self) -> bool:
        """
        Execute the appropriate ground truth extraction strategy for the dataset.

        This method delegates to the specialized extractor instance created
        during initialization.

        Returns
        -------
        bool
            True if extraction was successful, False otherwise.
        """
        if self._extractor is None:
            dataset_type = self._get_dataset_type()
            if dataset_type is None:
                logger.error('[%s] Unknown dataset type', self.dataset)
                print(f"Unknown dataset: {self.dataset}. Available choices: {', '.join(constants.LOGS_LIST)}")
            else:
                logger.error('[%s] No strategy found for type=%s', self.dataset, dataset_type)
                print(f"No extraction strategy found for dataset type: {dataset_type}")
            return False

        logger.info('[%s] Executing extraction strategy type=%s', self.dataset, self._get_dataset_type())
        self._extractor.extract()
        return True

