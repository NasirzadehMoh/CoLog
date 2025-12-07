"""
extraction/groundtruth_aggregator.py — GroundTruth aggregation

This module contains the GroundTruthAggregator class that combines groundtruths
from multiple datasets into a single aggregated groundtruth store.

This is used for cross-dataset experiments where training happens on combined
data from multiple sources.
"""

import pickle
from pathlib import Path
from .groundtruth_base import GroundTruthExtractor
from utils import DatasetAttributes
from utils import constants
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)


class GroundTruthAggregator(GroundTruthExtractor):
    """
    Aggregate groundtruths from multiple datasets.
    
    This class loads train/valid/test splits from several dataset-specific
    locations and merges dictionaries and embedding maps into combined
    pickles used for cross-dataset experiments.
    
    Aggregates: hadoop, zookeeper, bgl, casper-rw, dfrws-2009-jhuisi,
    dfrws-2009-nssal, honeynet-challenge7
    """

    def extract(self) -> None:
        """
        Combine groundtruths from multiple datasets.

        This helper loads train/valid/test splits from several dataset-specific
        locations and merges dictionaries and embedding maps into combined
        pickles.

        Returns
        -------
        None
        """
        all_train_keys = []
        all_train_messages = {}
        all_train_sequences = {}
        all_train_labels = {}

        all_valid_keys = []
        all_valid_messages = {}
        all_valid_sequences = {}
        all_valid_labels = {}

        all_test_keys = []
        all_test_messages = {}
        all_test_sequences = {}
        all_test_labels = {}

        all_embeddings = {}

        logger.info('[all] Aggregation start across datasets')
        datasets = ['hadoop', 'zookeeper', 'bgl', 'casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7']
        print("  - Checking for required files...")

        datasets_to_aggregate = []

        for dataset in datasets:
            train_set_dir = self.datasets_dir + dataset + '/' + constants.DIR_GROUNDTRUTH + f'{self.sequence_type.lower()}_{self.window_size}/' + f'train_valid_test_{self.train_ratio}' + '/' + constants.SUBDIR_RESAMPLED + f'/{self.sampling_method.lower()}/' + dataset + constants.SUFFIX_TRAIN_SET
            valid_set_dir = self.datasets_dir + dataset + '/' + constants.DIR_GROUNDTRUTH + f'{self.sequence_type.lower()}_{self.window_size}/' + f'train_valid_test_{self.train_ratio}' + '/' + constants.SUBDIR_RESAMPLED + f'/{self.sampling_method.lower()}/' + dataset + constants.SUFFIX_VALID_SET
            test_set_dir = self.datasets_dir + dataset + '/' + constants.DIR_GROUNDTRUTH + f'{self.sequence_type.lower()}_{self.window_size}/' + f'train_valid_test_{self.train_ratio}' + '/' + dataset + constants.SUFFIX_TEST_SET
            embs_dir = self.datasets_dir + dataset + '/' + constants.DIR_LOG_EMBEDDINGS + dataset + constants.SUFFIX_EMBEDDINGS
            
            train_set_path = Path(train_set_dir)
            valid_set_path = Path(valid_set_dir)
            test_set_path = Path(test_set_dir)
            embs_dir_path = Path(embs_dir)

            if train_set_path.exists() and valid_set_path.exists() and test_set_path.exists() and embs_dir_path.exists():
                datasets_to_aggregate.append(dataset)
            else:
                print(f"[all] Warning: Missing files for dataset '{dataset}' - skipping")
                logger.warning('[all] Missing one or more files for dataset=%s; skipping from aggregation', dataset)
        
        print()
        print(f"[all] Starting aggregation across {len(datasets_to_aggregate)} dataset(s): {', '.join(datasets_to_aggregate) if datasets_to_aggregate else '(none available)'}")
        if datasets_to_aggregate:
            print("    ✓ All required files found")
        print("  - Loading and merging datasets...")
        
        for dataset in tqdm(datasets_to_aggregate, total=len(datasets_to_aggregate), colour='#1b8079'):
            train_set_dir = self.datasets_dir + dataset + '/' + constants.DIR_GROUNDTRUTH + f'{self.sequence_type.lower()}_{self.window_size}/' + f'train_valid_test_{self.train_ratio}' + '/' + constants.SUBDIR_RESAMPLED + f'/{self.sampling_method.lower()}/' + dataset + constants.SUFFIX_TRAIN_SET
            valid_set_dir = self.datasets_dir + dataset + '/' + constants.DIR_GROUNDTRUTH + f'{self.sequence_type.lower()}_{self.window_size}/' + f'train_valid_test_{self.train_ratio}' + '/' + constants.SUBDIR_RESAMPLED + f'/{self.sampling_method.lower()}/' + dataset + constants.SUFFIX_VALID_SET
            test_set_dir = self.datasets_dir + dataset + '/' + constants.DIR_GROUNDTRUTH + f'{self.sequence_type.lower()}_{self.window_size}/' + f'train_valid_test_{self.train_ratio}' + '/' + dataset + constants.SUFFIX_TEST_SET
            embs_dir = self.datasets_dir + dataset + '/' + constants.DIR_LOG_EMBEDDINGS + dataset + constants.SUFFIX_EMBEDDINGS

            with open(train_set_dir, 'rb') as handle:
                train_set = pickle.load(handle)
            with open(valid_set_dir, 'rb') as handle:
                valid_set = pickle.load(handle)
            with open(test_set_dir, 'rb') as handle:
                test_set = pickle.load(handle)
            with open(embs_dir, 'rb') as handle:
                embs_dict = pickle.load(handle)

            all_train_keys = all_train_keys + train_set.keys
            all_train_messages.update(train_set.messages)
            all_train_sequences.update(train_set.sequences)
            all_train_labels.update(train_set.labels)

            all_valid_keys = all_valid_keys + valid_set.keys
            all_valid_messages.update(valid_set.messages)
            all_valid_sequences.update(valid_set.sequences)
            all_valid_labels.update(valid_set.labels)

            all_test_keys = all_test_keys + test_set.keys
            all_test_messages.update(test_set.messages)
            all_test_sequences.update(test_set.sequences)
            all_test_labels.update(test_set.labels)

            all_embeddings.update(embs_dict)

        print("  - Assembling final datasets...")
        all_train_set = DatasetAttributes(
            keys=all_train_keys,
            messages=all_train_messages,
            sequences=all_train_sequences,
            labels=all_train_labels
        )
        
        all_valid_set = DatasetAttributes(
            keys=all_valid_keys,
            messages=all_valid_messages,
            sequences=all_valid_sequences,
            labels=all_valid_labels
        )
        
        all_test_set = DatasetAttributes(
            keys=all_test_keys,
            messages=all_test_messages,
            sequences=all_test_sequences,
            labels=all_test_labels
        )

        # Count normal and anomaly samples
        train_normal = sum(1 for label in all_train_labels.values() if label == 0)
        train_anomaly = sum(1 for label in all_train_labels.values() if label == 1)
        valid_normal = sum(1 for label in all_valid_labels.values() if label == 0)
        valid_anomaly = sum(1 for label in all_valid_labels.values() if label == 1)
        test_normal = sum(1 for label in all_test_labels.values() if label == 0)
        test_anomaly = sum(1 for label in all_test_labels.values() if label == 1)
        
        print("\n  - Dataset composition:")
        print(f"    Train set: {train_normal} normal, {train_anomaly} anomaly")
        print(f"    Valid set: {valid_normal} normal, {valid_anomaly} anomaly")
        print(f"    Test set:  {test_normal} normal, {test_anomaly} anomaly")

        print("  - Saving aggregated files...")
        output_path = Path(self.datasets_dir) / 'all' / 'groundtruth' / f'{self.sequence_type.lower()}_{self.window_size}' / f'train_valid_test_{self.train_ratio}' / 'resampled_groundtruth' / f'{self.sampling_method.lower()}'
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"    - Saving train set ({len(all_train_keys)} items)...")
        with open(str(output_path / f'all{constants.SUFFIX_TRAIN_SET}'), 'wb') as handle:
            pickle.dump(all_train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del all_train_set

        print(f"    - Saving validation set ({len(all_valid_keys)} items)...")
        with open(str(output_path / f'all{constants.SUFFIX_VALID_SET}'), 'wb') as handle:
            pickle.dump(all_valid_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del all_valid_set

        print(f"    - Saving test set ({len(all_test_keys)} items)...")
        with open(str(Path(self.datasets_dir) / 'all' / 'groundtruth' / f'{self.sequence_type.lower()}_{self.window_size}' / f'train_valid_test_{self.train_ratio}' / f'all{constants.SUFFIX_TEST_SET}'), 'wb') as handle:
            pickle.dump(all_test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del all_test_set

        embeddings_path = self.datasets_dir + 'all/log_embeddings/'
        embeddings_path_obj = Path(embeddings_path)
        if not embeddings_path_obj.exists():
            embeddings_path_obj.mkdir(parents=True, exist_ok=True)

        print(f"    - Saving embeddings ({len(all_embeddings)} items)...")
        with open(embeddings_path + f'all{constants.SUFFIX_EMBEDDINGS}', 'wb') as handle:
            pickle.dump(all_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('[all] Aggregation complete: train=%d valid=%d test=%d embeddings=%d train_normal=%d train_anomaly=%d',
                    len(all_train_keys), len(all_valid_keys), len(all_test_keys), len(all_embeddings), train_normal, train_anomaly)
        print(f"\n[all] Assembling final datasets complete!")
        print(f"  - Train: {len(all_train_keys)} items ({train_normal} normal, {train_anomaly} anomaly)")
        print(f"  - Validation: {len(all_valid_keys)} items ({valid_normal} normal, {valid_anomaly} anomaly)")
        print(f"  - Test: {len(all_test_keys)} items ({test_normal} normal, {test_anomaly} anomaly)")
        print(f"  - Embeddings: {len(all_embeddings)} items")
        # Free memory used during aggregation
        try:
            del all_train_keys, all_train_messages, all_train_sequences, all_train_labels
            del all_valid_keys, all_valid_messages, all_valid_sequences, all_valid_labels
            del all_test_keys, all_test_messages, all_test_sequences, all_test_labels
            del all_embeddings
        except Exception as exc:
            logger.debug('[all] Failed to delete aggregated data from memory: %s', str(exc))
            print(f"all Warning: failed to delete aggregated data from memory: {str(exc)}")
        import gc
        gc.collect()
