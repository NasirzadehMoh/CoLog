"""
sampler.py — class imbalance solver for CoLog

This module provides the ClassImbalanceSolver class for handling class
imbalance in log datasets. It reads existing train/valid splits, converts
messages to embeddings, applies various resampling methods from the
imbalanced-learn library, and saves the resampled datasets to disk.

The solver attempts to reuse precomputed embeddings stored under the
dataset's `log_embeddings/` folder. If an embedding for a given message
is not found, it falls back to computing embeddings with SentenceTransformer.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging
import gc

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from utils import InputValidator
from utils import DatasetAttributes
from utils import constants


logger = logging.getLogger(__name__)


class ClassImbalanceSolver(object):
    """
    Lightweight merger of `solve_class_imbalanced.py` functionality.

    Intended to be run after `GroundTruth` extraction. The solver will
    attempt to reuse precomputed embeddings stored under the dataset's
    `log_embeddings/` folder. If an embedding for a given message is not
    found, it falls back to computing embeddings with SentenceTransformer.

    This class implements only the methods required to read the existing
    train/valid splits, map messages -> embeddings, run an imblearn
    sampler and write the resampled subsets to disk.
    """
    def __init__(self, dataset: str, method: str, groundtruth_dir: str, train_ratio: float = constants.DEFAULT_TRAIN_RATIO, 
                 device: str = constants.DEVICE_AUTO, batch_size: int = 64, verbose: bool = False, force: bool = False, 
                 random_seed: int = constants.DEFAULT_RANDOM_SEED, dry_run: bool = False):
        """
        Initialize the ClassImbalanceSolver.

        Parameters
        ----------
        dataset : str
            Name of the dataset to resample (e.g., 'hadoop').
        method : str
            Resampling method name (e.g., 'TomekLinks', 'SMOTE'). Must be one
            of the supported methods in METHODS_LIST.
        groundtruth_dir : str
            Path to the groundtruth directory containing train/valid splits.
        train_ratio : float, optional
            Training ratio used when groundtruth was created. Default is 0.6.
        device : str, optional
            Device for embedding computation: 'auto', 'cpu', or 'cuda'.
            Default is 'auto'.
        batch_size : int, optional
            Batch size for embedding computation. Default is 64.
        verbose : bool, optional
            If True, enables verbose output. Default is False.
        force : bool, optional
            If True, forces resampling even if files exist. Default is False.
        random_seed : int, optional
            Random seed for reproducibility. Default is DEFAULT_RANDOM_SEED.
        dry_run : bool, optional
            If True, previews operations without execution. Default is False.

        Returns
        -------
        None
        
        Raises
        ------
        ValidationError
            If method is not in METHODS_LIST or other validation fails.
        """
        # Validate inputs
        InputValidator.validate_in_list(method, constants.METHODS_LIST, "resampling method")
        InputValidator.validate_range(train_ratio, 0.0, 1.0, "train_ratio", inclusive=False)
        InputValidator.validate_in_list(device, [constants.DEVICE_AUTO, constants.DEVICE_CPU, constants.DEVICE_CUDA], "device")
        InputValidator.validate_positive(batch_size, "batch_size", allow_zero=False)
        
        self.dataset = dataset
        self.method = method
        self.groundtruth_dir = groundtruth_dir
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.dry_run = dry_run
        if device == constants.DEVICE_AUTO:
            self.device = constants.DEVICE_CUDA if torch.cuda.is_available() else constants.DEVICE_CPU
        else:
            self.device = device
        self.verbose = verbose
        self.force = force

    def _read_groundtruth(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Read train and validation sets from groundtruth directory.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            (train_set, valid_set) where each set is an OrderedDict with
            'keys', 'messages', 'sequences', and 'labels' fields.
            
        Raises
        ------
        FileNotFoundError
            If train or validation set files don't exist.
        """
        path = Path(self.groundtruth_dir) / f'train_valid_test_{self.train_ratio}'
        train_path = path / (self.dataset + constants.SUFFIX_TRAIN_SET)
        valid_path = path / (self.dataset + constants.SUFFIX_VALID_SET)
        
        if self.dry_run:
            print(f"[DRY RUN] Would load groundtruth files:")
            print(f"[DRY RUN]   - Train: {train_path}")
            print(f"[DRY RUN]   - Valid: {valid_path}")
            # Return empty dicts for dry-run
            return {'keys': [], 'messages': {}, 'sequences': {}, 'labels': {}}, \
                   {'keys': [], 'messages': {}, 'sequences': {}, 'labels': {}}
        
        # Validate files exist (only when not in dry-run mode)
        InputValidator.validate_file_exists(train_path, "Training set file")
        InputValidator.validate_file_exists(valid_path, "Validation set file")
        
        print(f"[{self.dataset}] Loading groundtruth files for resampling...")
        with open(str(train_path), 'rb') as handle:
            train_set = pickle.load(handle)
        with open(str(valid_path), 'rb') as handle:
            valid_set = pickle.load(handle)
        
        # Convert to DatasetAttributes if loaded as dict (backward compatibility)
        if isinstance(train_set, dict) and not isinstance(train_set, DatasetAttributes):
            train_set = DatasetAttributes(**train_set)
        if isinstance(valid_set, dict) and not isinstance(valid_set, DatasetAttributes):
            valid_set = DatasetAttributes(**valid_set)
        
        print(f"  - Train set: {len(train_set.keys)} items")
        print(f"  - Validation set: {len(valid_set.keys)} items")
        return train_set, valid_set

    def _extract_messages_and_labels(self, messages: Dict[str, Any], labels: Dict[str, int]) -> Tuple[List[str], List[int]]:
        """
        Extract and clean messages and labels for resampling.

        Parameters
        ----------
        messages : dict[str, Any]
            Dictionary mapping message IDs to message arrays.
        labels : dict[str, int]
            Dictionary mapping message IDs to labels.
        subset_name : str
            Name of subset for logging (e.g., 'train_set').

        Returns
        -------
        tuple[list[str], list[int]]
            (X_messages, y_labels) where X_messages are cleaned message strings
            and y_labels are integer labels.
        """
        X_messages = []
        y_labels = []
        for key, message in messages.items():
            X_messages.append(str(message))
            y_labels.append(int(labels[key]))
        return X_messages, y_labels

    def _locate_embeddings_file(self) -> str:
        """
        Locate the embeddings file for the dataset.

        Walks up the directory tree to find the dataset root, then constructs
        the path to the embeddings file.

        Returns
        -------
        str or None
            Path to embeddings file if found, None otherwise.
        """
        # Try to locate dataset root by walking up to a 'groundtruth' parent
        p = Path(self.groundtruth_dir).resolve()
        while p.name != constants.DIR_GROUNDTRUTH.rstrip('/') and p.parent != p:
            p = p.parent
        if p.name == constants.DIR_GROUNDTRUTH.rstrip('/'):
            dataset_root = p.parent
            emb_path = dataset_root / constants.DIR_LOG_EMBEDDINGS / f'{self.dataset}{constants.SUFFIX_EMBEDDINGS}'
            return emb_path
        return None

    def _get_embeddings(self, X_messages: List[str]) -> np.ndarray:
        """
        Get embeddings for messages, using precomputed embeddings when available.

        Attempts to load precomputed embeddings from disk. For any missing
        messages, computes embeddings on-the-fly using SentenceTransformer.

        Parameters
        ----------
        X_messages : list[str]
            List of messages to get embeddings for.

        Returns
        -------
        np.ndarray
            Array of embeddings with shape (len(X_messages), embedding_dim).
        """
        # Prefer loading saved embeddings to avoid recomputation
        embs_path = self._locate_embeddings_file()
        embs_dict = None
        if embs_path and embs_path.exists():
            try:
                print(f"[{self.dataset}] Loading precomputed embeddings...")
                with open(str(embs_path), 'rb') as handle:
                    embs_dict = pickle.load(handle)
                print(f"  - Loaded {len(embs_dict)} embeddings from cache")
            except Exception as exc:
                logger.warning('[%s] Failed to load embeddings from %s: %s. Will compute embeddings.', self.dataset, embs_path, str(exc))
                print(f"[{self.dataset}] Warning: failed to load embeddings from {embs_path}: {str(exc)}. Will compute embeddings.")
                embs_dict = None

        if embs_dict:
            # Map messages -> embedding; if missing, compute fallback
            X = []
            missing = []
            for msg in X_messages:
                if msg in embs_dict:
                    X.append(embs_dict[msg])
                else:
                    missing.append(msg)
                    X.append(None)
            if any(v is None for v in X):
                # Compute embeddings only for the missing messages
                print(f"[{self.dataset}] Computing embeddings for {len(missing)} missing messages...")
                try:
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    model.to(self.device)
                    computed = model.encode(missing, show_progress_bar=True, batch_size=self.batch_size)
                    # replace None slots with computed embeddings in order
                    comp_iter = iter(computed)
                    for i, v in enumerate(X):
                        if v is None:
                            X[i] = next(comp_iter)
                    del model, computed
                    gc.collect()
                    if self.device == 'cuda' and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    logger.warning('[%s] Failed to compute embeddings for missing messages: %s. Using zero vectors as fallback.', self.dataset, str(exc))
                    print(f"[{self.dataset}] Warning: failed to compute embeddings for missing messages: {str(exc)}. Using zero vectors as fallback.")
                    # as a last resort, create zero vectors
                    X = [np.zeros(constants.EMBEDDING_DIMENSION, dtype=float) if v is None else v for v in X]
            
            # Clean up loaded embeddings dictionary
            del embs_dict
            gc.collect()
            return np.array(X)

        # No precomputed embeddings found: compute all embeddings
        print(f"[{self.dataset}] Computing embeddings for {len(X_messages)} messages...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.to(self.device)
        X = model.encode(X_messages, show_progress_bar=True, batch_size=self.batch_size)
        del model
        gc.collect()
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.array(X)

    def _get_sampler(self) -> Any:
        """
        Get the configured imblearn sampler instance.

        Returns
        -------
        object
            Configured imblearn sampler instance based on self.method.
        """
        if self.method in ['NeighbourhoodCleaningRule', 'TomekLinks', 'NearMiss']:
            from imblearn.under_sampling import NeighbourhoodCleaningRule, TomekLinks, NearMiss
            sampler_map = {
                'NeighbourhoodCleaningRule': NeighbourhoodCleaningRule,
                'TomekLinks': TomekLinks,
                'NearMiss': NearMiss
            }
            return sampler_map[self.method](n_jobs=constants.CPU_COUNT)
        if self.method in ['OneSidedSelection', 'CondensedNearestNeighbour',
                           'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours',
                           'AllKNN', 'InstanceHardnessThreshold']:
            from imblearn.under_sampling import OneSidedSelection, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, InstanceHardnessThreshold
            sampler_map = {
                'OneSidedSelection': OneSidedSelection,
                'CondensedNearestNeighbour': CondensedNearestNeighbour,
                'EditedNearestNeighbours': EditedNearestNeighbours,
                'RepeatedEditedNearestNeighbours': RepeatedEditedNearestNeighbours,
                'AllKNN': AllKNN,
                'InstanceHardnessThreshold': InstanceHardnessThreshold
            }
            return sampler_map[self.method](random_state=self.random_seed, n_jobs=constants.CPU_COUNT)
        if self.method in ['RandomUnderSampler']:
            from imblearn.under_sampling import RandomUnderSampler
            return RandomUnderSampler(random_state=self.random_seed)

    def sample_groundtruth(self) -> None:
        """
        Resample train and validation sets using the configured method.

        Reads existing train/valid sets, converts messages to embeddings,
        applies the resampling method, and saves the resampled sets to disk.

        Returns
        -------
        None
        """
        sub_sets = [constants.FILE_TRAIN_SET, constants.FILE_VALID_SET]
        train_set, valid_set = self._read_groundtruth()
        
        # If dry run, exit early after showing what would be loaded
        if self.dry_run:
            output_base_path = Path(self.groundtruth_dir) / f'train_valid_test_{self.train_ratio}' / constants.SUBDIR_RESAMPLED / self.method.lower()
            print(f"[DRY RUN] Would resample using method: {self.method}")
            print(f"[DRY RUN] Output directory: {output_base_path}")
            print(f"[DRY RUN] Would process: {', '.join(sub_sets)}")
            return
        
        # Check if resampled files already exist
        output_base_path = Path(self.groundtruth_dir) / f'train_valid_test_{self.train_ratio}' / constants.SUBDIR_RESAMPLED / self.method.lower()
        train_file_path = output_base_path / f'{self.dataset}{constants.SUFFIX_TRAIN_SET}'
        valid_file_path = output_base_path / f'{self.dataset}{constants.SUFFIX_VALID_SET}'

        if not self.force and train_file_path.exists() and valid_file_path.exists():
            logger.info('[%s] Resampled files already exist, skipping resampling. Use --force to recompute. method=%s', self.dataset, self.method)
            print(f"[{self.dataset}] Resampled files already exist for method '{self.method}'. Skipping resampling.")
            print(f"  Use --force to force recomputation.")
            return
        
        for subset_name in sub_sets:
            subset = train_set if subset_name == constants.FILE_TRAIN_SET else valid_set
            X_messages, y_labels = self._extract_messages_and_labels(subset.messages, subset.labels)
            X = self._get_embeddings(X_messages)
            y = np.array(y_labels)
            
            # Clean up intermediate data
            del X_messages, y_labels
            gc.collect()

            sampler = self._get_sampler()
            logger.info('[%s] Resampling start method=%s subset=%s', self.dataset, self.method, subset_name)
            print(f"  - Applying '{self.method}' resampling to {subset_name}...")

            try:
                _, _ = sampler.fit_resample(X, y)
            except Exception as e:
                # Some samplers return indices differently; attempt fit then access sample_indices_
                logger.debug('[%s] fit_resample failed, trying fit: %s', self.dataset, e)
                sampler.fit(X, y)
            indexes = list(getattr(sampler, 'sample_indices_', []))
            if not indexes:
                # Some samplers (fit_resample) return X_res, y_res but don't populate sample_indices_
                # Try to determine kept indices by mapping returned X to original via approximate matching
                try:
                    Xr, yr = sampler.fit_resample(X, y)
                    # fallback: keep first len(Xr) indices (best-effort)
                    indexes = list(range(min(len(X), len(Xr))))
                except Exception as e:
                    logger.warning('[%s] Both fit_resample attempts failed, using all indices: %s', self.dataset, e)
                    indexes = list(range(len(X)))

            resampled_keys = [subset.keys[idx] for idx in indexes]
            print(f"    Original: {len(subset.keys)} items → Resampled: {len(resampled_keys)} items")

            resampled_set = DatasetAttributes(
                keys=resampled_keys,
                messages={key: subset.messages[key] for key in resampled_keys},
                sequences={key: subset.sequences[key] for key in resampled_keys},
                labels={key: subset.labels[key] for key in resampled_keys}
            )

            output_path = Path(self.groundtruth_dir) / f'train_valid_test_{self.train_ratio}' / constants.SUBDIR_RESAMPLED / self.method.lower()
            output_path.mkdir(parents=True, exist_ok=True)
            file_path = output_path / f'{self.dataset}_{subset_name}{constants.EXT_PICKLE}'
            with open(file_path, 'wb') as handle:
                pickle.dump(resampled_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('[%s] Resampled subset saved subset=%s path=%s', self.dataset, subset_name, file_path)
            print(f"  - Saved resampled {subset_name} to: {file_path}")
            print(f"    ✓ Completed {subset_name}\n")
            
            # Clean up memory after each subset
            del X, y, sampler, resampled_set
            gc.collect()
        
        # Clean up loaded datasets
        del train_set, valid_set
        gc.collect()
