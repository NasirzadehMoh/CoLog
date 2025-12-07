"""
extraction/groundtruth_base.py — Base class for GroundTruth extraction

This module contains the base GroundTruthExtractor class with shared methods
used across all dataset types for sequence building, labeling, saving, and
dataset splitting.
"""

import pickle
from pathlib import Path
from utils import ISettingsProvider, SettingsAdapter
from utils import DatasetAttributes
from utils import (
    ValidationError,
    FileNotFoundError,
    ConfigurationError
)
from utils import InputValidator
from utils import constants
import re
from datetime import datetime
import numpy as np
import random
from math import floor
import logging
from typing import List, Dict, Tuple, Any
import gc


logger = logging.getLogger(__name__)


class GroundTruthExtractor(object):
    """
    Base class for ground truth extraction across different dataset types.

    This class contains all shared functionality for reading parsed logs,
    constructing sequences, assigning labels, and writing ground truth files.
    Subclasses implement dataset-specific extraction logic.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'hadoop').
    sequence_type : str
        'background' or 'context' to control sequence building.
    window_size : int
        Number of messages on each side. Must be non-negative.
    train_ratio : float
        Fraction for training set. Must be in (0, 1).
    valid_ratio : float
        Fraction for validation set. Must be in (0, 1).
    sampling_method : str
        Sampling method name.
    datasets_dir : str, optional
        Root directory containing datasets. Default is 'datasets/'.
    force : bool, optional
        Force re-extraction if True. Default is False.
    groundbreaking : bool, optional
        Enable multi-label mode if True. Default is False.
    random_seed : int, optional
        Random seed for reproducibility. Default is 100.
    dry_run : bool, optional
        Preview operations without execution if True. Default is False.
    settings_provider : ISettingsProvider, optional
        Settings provider for dependency injection. If None, creates a
        SettingsAdapter. Default is None.
    """
    
    def __init__(self, dataset: str, sequence_type: str, window_size: int, train_ratio: float, 
                 valid_ratio: float, sampling_method: str, datasets_dir: str = 'datasets/', 
                 force: bool = False, groundbreaking: bool = False, 
                 random_seed: int = constants.DEFAULT_RANDOM_SEED, dry_run: bool = False, 
                 settings_provider: ISettingsProvider = None) -> None:
        """Initialize GroundTruthExtractor with validation."""
        
        # Validate inputs
        InputValidator.validate_in_list(sequence_type, [constants.SEQ_TYPE_BACKGROUND, constants.SEQ_TYPE_CONTEXT], "sequence_type")
        InputValidator.validate_positive(window_size, "window_size", allow_zero=True)
        InputValidator.validate_range(train_ratio, 0.0, 1.0, "train_ratio", inclusive=False)
        InputValidator.validate_range(valid_ratio, 0.0, 1.0, "valid_ratio", inclusive=False)
        
        # Validate that train_ratio + valid_ratio < 1.0
        if train_ratio + valid_ratio >= 1.0:
            raise ValidationError(
                f"Sum of train_ratio ({train_ratio}) and valid_ratio ({valid_ratio}) "
                f"must be less than 1.0"
            )
        
        self.dataset = dataset
        self.sequence_type = sequence_type
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.sampling_method = sampling_method
        self.random_seed = random_seed
        self.dry_run = dry_run
        
        # Use dependency injection for settings
        if settings_provider is None:
            self.settings_provider = SettingsAdapter(dataset)
        else:
            self.settings_provider = settings_provider
        self.settings = self.settings_provider.get_settings()
        
        # Configuration
        self.datasets_dir = datasets_dir
        self.force = force
        self.groundbreaking = groundbreaking

    @staticmethod
    def _read_wordlist(log_type: str) -> List[str]:
        """
        Read the anomaly keyword list for the given log_type.

        Parameters
        ----------
        log_type : str
            The type of log (e.g., 'spark', 'syslog').

        Returns
        -------
        list[str]
            A list of lowercase tokens used for anomaly detection.
        """
        wordlist_root_path = Path(constants.DIR_GROUNDTRUTH) / constants.DIR_WORDLIST
        wordlist_path = wordlist_root_path / (log_type + constants.EXT_TXT)
        
        if not wordlist_path.exists():
            logger.warning('Wordlist not found for %s at %s; proceeding with empty wordlist', log_type, wordlist_path)
            return []
        
        try:
            with open(str(wordlist_path), 'r') as f:
                wordlist_lines = f.readlines()
        except Exception as exc:
            logger.error('Failed to read wordlist for %s at %s: %s', log_type, wordlist_path, str(exc))
            print(f"{log_type} Warning: failed to read wordlist from {wordlist_path}: {str(exc)}")
            return []

        wordlist = []
        for word in wordlist_lines:
            wordlist.append(word.strip().lower())

        return wordlist

    @staticmethod
    def _set_anomaly_label(wordlist: List[str], message: str) -> int:
        """
        Check whether any token from wordlist is present in message.

        Parameters
        ----------
        wordlist : list[str]
            List of lowercase anomaly keywords to search for.
        message : str
            The log message to check.

        Returns
        -------
        int
            0 if any wordlist token appears in message (anomaly),
            1 otherwise (normal).
        """
        message = message.lower().strip()
        anomaly_label = 1

        for word in wordlist:
            if word in message:
                anomaly_label = 0
                return anomaly_label

        return anomaly_label

    @staticmethod
    def _clean_message(message: str) -> List[str]:
        """
        Normalize and clean a message string.

        Parameters
        ----------
        message : str
            The log message to clean.

        Returns
        -------
        list[str]
            List of cleaned tokens from the message.
        """
        if not isinstance(message, str):
            message = str(message)
        message = re.sub(r"([.,'!?\"()*#:;])", '', message)
        message = message.replace('=', ' ').replace('/', ' ').replace('-', ' ').split()

        return message

    def _build_sequence(self, messages_list: List[str], index: int, window_size: int, sequence_type: str) -> Tuple[List[int], List[str]]:
        """
        Build a sequence around a message at index.

        Parameters
        ----------
        messages_list : list[str]
            List of messages in the file (ordered).
        index : int
            Index of the message to center the sequence on.
        window_size : int
            Number of messages to include on each side.
        sequence_type : str
            'background' (left+current) or 'context' (left+current+right).

        Returns
        -------
        tuple[list[int], list[str]]
            A tuple (indices, sequence_tokens).
        """
        seq: List[str] = []
        # Left side
        left_start = max(0, index - window_size)
        left = messages_list[left_start:index]
        left_unk_count = window_size - len(left)
        seq.extend([constants.UNKNOWN_TOKEN] * left_unk_count)
        seq.extend(left)
        indices = list(range(left_start, index))
        # Current message
        if not self.groundbreaking:
            seq.append(messages_list[index])
            indices.extend([index])
        # Right side (optional)
        if sequence_type == constants.SEQ_TYPE_CONTEXT:
            right_end = min(len(messages_list), index + window_size + 1)
            right = messages_list[index + 1:right_end]
            seq.extend(right)
            right_unk_count = window_size - len(right)
            seq.extend([constants.UNKNOWN_TOKEN] * right_unk_count)
            indices.extend(list(range(index + 1, right_end)))
        return indices, seq

    def _save_groundtruth(self, groundtruth_file: Any, groundtruth_type: str) -> None:
        """
        Persist a pickled groundtruth object to disk.

        Parameters
        ----------
        groundtruth_file : object
            Python object to pickle.
        groundtruth_type : str
            Controls the filename and folder.

        Returns
        -------
        None
        """
        output_path = Path(self.datasets_dir) / self.settings['groundtruth_dir'] / f"{self.sequence_type}_{self.window_size}"
        
        if groundtruth_type not in [constants.FILE_TRAIN_SET, constants.FILE_VALID_SET, constants.FILE_TEST_SET, constants.FILE_GENERALIZATION]:
            output_path = output_path / constants.SUBDIR_ALL
            groundtruth_path = output_path / (self.dataset + '_' + groundtruth_type + constants.EXT_PICKLE)
        elif groundtruth_type == constants.FILE_GENERALIZATION:
            groundtruth_path = output_path / (self.dataset + constants.EXT_PICKLE)
        else:
            output_path = output_path / f'train_valid_test_{self.train_ratio}'
            groundtruth_path = output_path / (self.dataset + '_' + groundtruth_type + constants.EXT_PICKLE)
        
        if self.dry_run:
            print(f"[DRY RUN] Would save {groundtruth_type} to: {groundtruth_path}")
            if isinstance(groundtruth_file, dict):
                print(f"[DRY RUN] Object contains {len(groundtruth_file)} items")
            elif isinstance(groundtruth_file, list):
                print(f"[DRY RUN] Object contains {len(groundtruth_file)} items")
            return

        if groundtruth_path.exists() and not self.force:
            logger.info('[%s] Groundtruth file exists and --force not set; skipping write: %s', self.dataset, groundtruth_path)
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(str(groundtruth_path), 'wb') as handle:
            pickle.dump(groundtruth_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info('[%s] Saved groundtruth object type=%s path=%s', self.dataset, groundtruth_type, groundtruth_path)

    def _split_dataset(self, messages: Dict[str, Any], sequences: Dict[str, Any], labels: Dict[str, int]) -> Tuple[List[str], List[str], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Create train/validation/test splits.

        Parameters
        ----------
        messages : dict[str, Any]
            Dictionary mapping message IDs to message arrays.
        sequences : dict[str, Any]
            Dictionary mapping message IDs to sequence lists.
        labels : dict[str, int]
            Dictionary mapping message IDs to labels.

        Returns
        -------
        tuple
            (normal_indexes, anomaly_indexes, train_set, valid_set, test_set)
        """
        normal_indexes = []
        anomaly_indexes = []
        for index, label in labels.items():
            # For groundbreaking mode: 0,1,2=anomaly, 3=normal
            # For normal mode: 0=anomaly, 1=normal
            if self.groundbreaking:
                if label == 3:  # Message and sequence normal
                    normal_indexes.append(index)
                elif label in [0, 1, 2]:  # Message, sequence or both anomaly (regardless of sequence context)
                    anomaly_indexes.append(index)
            else:
                if label == 1: # Normal message
                    normal_indexes.append(index)
                elif label == 0: # Anomalous message
                    anomaly_indexes.append(index)
        
        rng = random.Random(self.random_seed)
        rng.shuffle(normal_indexes)
        rng.shuffle(anomaly_indexes)

        len_normal = len(normal_indexes)
        len_anomaly = len(anomaly_indexes)

        len_train_normal = floor(self.train_ratio * len_normal)
        len_val_normal = floor(self.valid_ratio * len_normal)
        len_train_anomaly = floor(self.train_ratio * len_anomaly)
        len_val_anomaly = floor(self.valid_ratio * len_anomaly)

        train_indexes = normal_indexes[:len_train_normal] + anomaly_indexes[:len_train_anomaly]
        valid_indexes = normal_indexes[len_train_normal:len_train_normal+len_val_normal] + anomaly_indexes[len_train_anomaly:len_train_anomaly+len_val_anomaly]
        test_indexes = normal_indexes[len_train_normal+len_val_normal:] + anomaly_indexes[len_train_anomaly+len_val_anomaly:]

        train_set = DatasetAttributes(
            keys=train_indexes,
            messages={key: messages[key] for key in train_indexes},
            sequences={key: sequences[key] for key in train_indexes},
            labels={key: labels[key] for key in train_indexes}
        )
        
        valid_set = DatasetAttributes(
            keys=valid_indexes,
            messages={key: messages[key] for key in valid_indexes},
            sequences={key: sequences[key] for key in valid_indexes},
            labels={key: labels[key] for key in valid_indexes}
        )
        
        test_set = DatasetAttributes(
            keys=test_indexes,
            messages={key: messages[key] for key in test_indexes},
            sequences={key: sequences[key] for key in test_indexes},
            labels={key: labels[key] for key in test_indexes}
        )

        logger.info('[%s] Dataset split completed: normal=%d anomaly=%d train=%d valid=%d test=%d',
                self.dataset,
                len(normal_indexes), len(anomaly_indexes),
                len(train_set.keys), len(valid_set.keys), len(test_set.keys))
        return normal_indexes, anomaly_indexes, train_set, valid_set, test_set
    
    @staticmethod
    def _print_counts_table(normal_count: int, anomaly_count: int) -> None:
        """
        Print a small table showing normal and anomaly counts.

        Parameters
        ----------
        normal_count : int
            Number of normal messages.
        anomaly_count : int
            Number of anomalous messages.

        Returns
        -------
        None
        """
        from tabulate import tabulate
        
        total = normal_count + anomaly_count
        normal_pct = (normal_count / total * 100) if total > 0 else 0
        anomaly_pct = (anomaly_count / total * 100) if total > 0 else 0
        
        table_data = [
            ['Normal', normal_count, f'{normal_pct:.2f}%'],
            ['Anomaly', anomaly_count, f'{anomaly_pct:.2f}%'],
            ['Total', total, '100.00%']
        ]
        print()
        print(tabulate(table_data, headers=['Category', 'Count', 'Percentage'], tablefmt='fancy_grid'))

    def _validate_and_check_paths(self) -> Tuple[bool, str]:
        """
        Validate settings and check if groundtruth extraction should proceed.
        
        Returns
        -------
        tuple[bool, str or None]
            (should_continue, logs_dir)
        """
        required_keys = ['out_dir', 'groundtruth_dir']
        if not InputValidator.validate_settings_dict(self.settings, required_keys, self.dataset):
            if not self.dry_run:
                raise ConfigurationError(
                    f"[{self.dataset}] Missing required settings: "
                    f"{', '.join([k for k in required_keys if k not in self.settings])}"
                )
            return False, None

        logs_dir = Path(self.datasets_dir) / self.settings['out_dir']
        base_output = Path(self.datasets_dir) / self.settings['groundtruth_dir'] / f"{self.sequence_type}_{self.window_size}"
        
        # Check all required groundtruth files in the 'all' subdirectory
        all_subdir = base_output / constants.SUBDIR_ALL
        messages_path = all_subdir / f"{self.dataset}_{constants.FILE_MESSAGES}{constants.EXT_PICKLE}"
        sequences_path = all_subdir / f"{self.dataset}_{constants.FILE_SEQUENCES}{constants.EXT_PICKLE}"
        labels_path = all_subdir / f"{self.dataset}_{constants.FILE_LABELS}{constants.EXT_PICKLE}"
        keys_path = all_subdir / f"{self.dataset}_{constants.FILE_KEYS}{constants.EXT_PICKLE}"
        
        # Required files always include messages, sequences, labels, keys
        required_files = [messages_path, sequences_path, labels_path, keys_path]
        
        # Determine if this dataset uses train/valid/test split or generalization set
        # Windows and honeynet-challenge5 use generalization, all others use train/valid/test
        uses_generalization = self.dataset in ['windows', 'honeynet-challenge5']
        
        if uses_generalization:
            # Check for generalization test set
            generalization_path = base_output / f"{self.dataset}{constants.EXT_PICKLE}"
            required_files.append(generalization_path)
        else:
            # Check train/valid/test split files
            split_subdir = base_output / f'train_valid_test_{self.train_ratio}'
            train_path = split_subdir / f"{self.dataset}_{constants.FILE_TRAIN_SET}{constants.EXT_PICKLE}"
            valid_path = split_subdir / f"{self.dataset}_{constants.FILE_VALID_SET}{constants.EXT_PICKLE}"
            test_path = split_subdir / f"{self.dataset}_{constants.FILE_TEST_SET}{constants.EXT_PICKLE}"
            required_files.extend([train_path, valid_path, test_path])
        
        # Check if all required files exist
        all_files_exist = all(f.exists() for f in required_files)
        
        if self.dry_run:
            print(f"[DRY RUN] Would check logs directory: {logs_dir}")
            print(f"[DRY RUN] Would check existing groundtruth files:")
            for filepath in required_files:
                exists_str = "EXISTS" if filepath.exists() else "MISSING"
                print(f"  [{exists_str}] {filepath}")
            if not logs_dir.exists():
                print(f"[DRY RUN] WARNING: Logs directory does not exist: {logs_dir}")
                print(f"[DRY RUN] Please ensure preprocessing has been run successfully.")
                return False, None
            if all_files_exist and not self.force:
                print(f"[DRY RUN] NOTE: All groundtruth files exist (would skip unless --force)")
                return False, None
            # If --force is set or files don't exist, continue with preview
            return True, logs_dir
        
        # Validate logs directory exists first
        try:
            InputValidator.validate_directory_exists(logs_dir, f"Structured logs directory for {self.dataset}")
        except FileNotFoundError as e:
            logger.error('[%s] %s', self.dataset, str(e))
            print(str(e))
            return False, None
        
        # Then check if all groundtruth files already exist
        if all_files_exist and not self.force:
            logger.info('[%s] All groundtruth files already exist and --force not set; skipping extraction', self.dataset)
            print(f"Skipping groundtruth extraction for {self.dataset}: all groundtruth files already exist")
            return False, None
        
        # If some files are missing, log which ones
        if not all_files_exist:
            missing_files = [str(f) for f in required_files if not f.exists()]
            logger.info('[%s] Some groundtruth files missing (%d/%d), proceeding with extraction', 
                       self.dataset, len(missing_files), len(required_files))
            if missing_files:
                logger.debug('[%s] Missing files: %s', self.dataset, ', '.join(missing_files))
        
        return True, logs_dir

    def _compute_groundbreaking_label(self, message_label: int, sequence_label: int) -> int:
        """
        Compute groundbreaking label from message and sequence labels.
        
        Parameters
        ----------
        message_label : int
            Label for the current message (0=anomaly, 1=normal).
        sequence_label : int
            Label for the sequence context (0=anomaly, 1=normal).
        
        Returns
        -------
        int
            Multi-class label (0-3).
        """
        if message_label == 0 and sequence_label == 0:
            return 0
        elif message_label == 0 and sequence_label == 1:
            return 1
        elif message_label == 1 and sequence_label == 0:
            return 2
        else:  # message_label == 1 and sequence_label == 1
            return 3

    def _save_all_groundtruth_files(self, messages: Dict[str, np.ndarray], sequences: Dict[str, List[str]], labels: Dict[str, int], keys: List[str]) -> None:
        """
        Save all groundtruth files (messages, sequences, labels, keys).

        Parameters
        ----------
        messages : dict[str, np.ndarray]
            Dictionary mapping message IDs to message token arrays.
        sequences : dict[str, list[str]]
            Dictionary mapping message IDs to sequence lists.
        labels : dict[str, int]
            Dictionary mapping message IDs to labels.
        keys : list[str]
            Ordered list of all message IDs.

        Returns
        -------
        None
        """
        print(f"\nPlease wait to save {self.dataset} dataset's ground truth.")
        start_time = datetime.now()
        self._save_groundtruth(messages, constants.FILE_MESSAGES)
        self._save_groundtruth(sequences, constants.FILE_SEQUENCES)
        self._save_groundtruth(labels, constants.FILE_LABELS)
        self._save_groundtruth(keys, constants.FILE_KEYS)
        print(f'Saving ground truth done. [Time taken: {datetime.now() - start_time}]\n')

    def _split_and_save_train_valid_test(self, messages: Dict[str, np.ndarray], sequences: Dict[str, List[str]], labels: Dict[str, int], type_name: str) -> None:
        """
        Split dataset into train/valid/test sets and save them.

        Parameters
        ----------
        messages : dict[str, np.ndarray]
            Dictionary mapping message IDs to message token arrays.
        sequences : dict[str, list[str]]
            Dictionary mapping message IDs to sequence lists.
        labels : dict[str, int]
            Dictionary mapping message IDs to labels.
        type_name : str
            Name of the dataset type for logging.

        Returns
        -------
        None
        """
        print(f"[{self.dataset}] Splitting dataset into train/valid/test sets...")
        start_time = datetime.now()
        normal_indexes, anomaly_indexes, train_set, valid_set, test_set = self._split_dataset(messages, sequences, labels)
        self._print_counts_table(len(normal_indexes), len(anomaly_indexes))
        logger.info('[%s] %s counts normal=%d anomaly=%d', self.dataset, type_name, len(normal_indexes), len(anomaly_indexes))
        
        print(f"  - Saving train set ({len(train_set.keys)} items)...")
        self._save_groundtruth(train_set, constants.FILE_TRAIN_SET)
        print(f"  - Saving validation set ({len(valid_set.keys)} items)...")
        self._save_groundtruth(valid_set, constants.FILE_VALID_SET)
        print(f"  - Saving test set ({len(test_set.keys)} items)...")
        self._save_groundtruth(test_set, constants.FILE_TEST_SET)
        print(f'[{self.dataset}] Splitting ground truth done. [Time taken: {datetime.now() - start_time}]\n')
        
        # Free memory
        try:
            del messages, sequences, labels, train_set, valid_set, test_set, normal_indexes, anomaly_indexes
        except Exception as exc:
            logger.debug('[%s] Failed to delete split datasets from memory: %s', self.dataset, str(exc))
            print(f"{self.dataset} Warning: failed to delete split datasets from memory: {str(exc)}")
        gc.collect()

    def _save_generalization_set(self, messages: Dict[str, np.ndarray], sequences: Dict[str, List[str]], labels: Dict[str, int], keys: List[str], type_name: str) -> None:
        """
        Save generalization test set.

        Parameters
        ----------
        messages : dict[str, np.ndarray]
            Dictionary mapping message IDs to message token arrays.
        sequences : dict[str, list[str]]
            Dictionary mapping message IDs to sequence lists.
        labels : dict[str, int]
            Dictionary mapping message IDs to labels.
        keys : list[str]
            Ordered list of all message IDs.
        type_name : str
            Name of the dataset type for logging.

        Returns
        -------
        None
        """
        print(f"[{self.dataset}] Preparing generalization test set...")
        start_time = datetime.now()
        
        test = DatasetAttributes(
            keys=keys,
            messages=messages,
            sequences=sequences,
            labels=labels
        )

        labels_list = list(labels.values())
        if self.groundbreaking:
            # In groundbreaking mode: 3=normal, 0/1/2=anomaly
            normal_indexes_count = labels_list.count(3)
            anomaly_indexes_count = sum(labels_list.count(label) for label in (0, 1, 2))
        else:
            # In standard mode: 1=normal, 0=anomaly
            normal_indexes_count = labels_list.count(1)
            anomaly_indexes_count = labels_list.count(0)

        self._print_counts_table(normal_indexes_count, anomaly_indexes_count)
        logger.info('[%s] %s generalization counts normal=%d anomaly=%d', self.dataset, type_name, normal_indexes_count, anomaly_indexes_count)
        print(f"  - Saving generalization set ({len(keys)} items)...")
        self._save_groundtruth(test, constants.FILE_GENERALIZATION)
        print(f'[{self.dataset}] Saving test ground truth done. [Time taken: {datetime.now() - start_time}]\n')
        
        # Free memory
        try:
            del messages, sequences, labels, keys, test
        except Exception as exc:
            logger.debug('[%s] Failed to delete generalization set from memory: %s', self.dataset, str(exc))
            print(f"{self.dataset} Warning: failed to delete generalization set from memory: {str(exc)}")
        gc.collect()

    def extract(self) -> None:
        """
        Extract ground truth for the dataset.
        
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement extract() method")
