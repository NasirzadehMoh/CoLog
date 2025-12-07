"""
preprocessor.py — Preprocessing functionality for CoLog datasets

This module provides the Preprocessor class which handles log parsing
and embedding computation for CoLog datasets. It supports two parsing
strategies:
    - Drain parser: for structured log parsing
    - NER parser: for Named Entity Recognition-based parsing

The Preprocessor computes embeddings using SentenceTransformer models
and saves them for later use in ground truth extraction.

Dependencies: pandas, sentence-transformers, torch, pickle
"""

import pickle
import gc
import logging
from pathlib import Path
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

from utils import InputValidator
from utils import ISettingsProvider, SettingsAdapter
from utils import constants

# module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class Preprocessor(object):
    """
    Minimal merged Preprocessor class (extracted from preprocess.py).

    This class provides only the pieces required by `groundtruth.py`:
      - collect and parse files with Drain or NER parsers
      - compute and save embeddings using SentenceTransformer

    It intentionally omits CLI parsing and logging configuration (those are
    already handled by `groundtruth.py`). The public API is `run()` which
    returns True on success, False on failure.
    """
    def __init__(self, dataset: str, dataset_dir: str, model: str, batch_size: int, device: str, 
                 force: bool, verbose: bool = False, dry_run: bool = False, 
                 settings_provider: ISettingsProvider = None) -> None:
        """
        Initialize the Preprocessor.

        Parameters
        ----------
        dataset : str
            Name of the dataset to preprocess (e.g., 'hadoop', 'casper-rw').
        dataset_dir : str
            Root directory containing all datasets.
        model : str
            Name of the SentenceTransformer model to use for embeddings
            (e.g., 'all-MiniLM-L6-v2').
        batch_size : int
            Batch size for embedding computation. Larger values use more memory
            but may be faster.
        device : str
            Device for computation: 'auto', 'cpu', or 'cuda'. 'auto' selects
            'cuda' if available, otherwise 'cpu'.
        force : bool
            If True, forces re-parsing and re-computation of embeddings even if
            files already exist.
        verbose : bool, optional
            If True, enables verbose output. Default is False.
        dry_run : bool, optional
            If True, previews operations without executing them. Default is False.
        settings_provider : ISettingsProvider, optional
            Settings provider for dependency injection. If None, creates a
            SettingsAdapter using the Settings class. Default is None.

        Returns
        -------
        None
        
        Raises
        ------
        ValidationError
            If batch_size is not positive.
        """
        # Validate inputs
        InputValidator.validate_positive(batch_size, "batch_size", allow_zero=False)
        InputValidator.validate_in_list(device, [constants.DEVICE_AUTO, constants.DEVICE_CPU, constants.DEVICE_CUDA], "device")
        
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.model = model
        self.batch_size = batch_size
        # Normalize device string: if 'auto', pick 'cuda' when available, otherwise 'cpu'.
        if device == constants.DEVICE_AUTO:
            self.device = constants.DEVICE_CUDA if torch.cuda.is_available() else constants.DEVICE_CPU
        else:
            self.device = device
        self.force = force
        self.verbose = verbose
        self.dry_run = dry_run
        # Use dependency injection for settings
        if settings_provider is None:
            self.settings_provider = SettingsAdapter(dataset)
        else:
            self.settings_provider = settings_provider
        self.settings = self.settings_provider.get_settings()

    def _compute_and_save_embeddings(self, messages_list: List[str], embs_file: str) -> bool:
        """
        Compute and save embeddings for a list of messages.

        Parameters
        ----------
        messages_list : list[str]
            List of unique messages to compute embeddings for.
        embs_file : str
            Path to save the embeddings pickle file.

        Returns
        -------
        bool
            True if embeddings were successfully computed and saved (or already
            exist and --force not set), False on failure.
        """
        if self.dry_run:
            print(f"[DRY RUN] Would compute embeddings for {len(messages_list)} messages")
            print(f"[DRY RUN] Would save to: {embs_file}")
            print(f"[DRY RUN] Model: {self.model}, Device: {self.device}, Batch size: {self.batch_size}")
            return True
        
        embs_path = Path(embs_file)
        if embs_path.exists() and not self.force:
            logger.info('[%s] Embeddings exist and --force not set: %s', self.dataset, embs_file)
            if self.verbose:
                print(f'{self.dataset} Embeddings already exist: {embs_file}')
            return True

        try:
            model = SentenceTransformer(self.model)
        except Exception as exc:
            logger.exception('[%s] Failed to load model %s: %s', self.dataset, self.model, str(exc))
            return False
        model.to(self.device)

        try:
            if self.verbose:
                print(f'{self.dataset} Computing embeddings (model={self.model}, device={self.device}, batch_size={self.batch_size})')
            embs = model.encode(messages_list, show_progress_bar=True, batch_size=self.batch_size)
            embs_dict = dict(zip(messages_list, embs))

            Path(embs_file).parent.mkdir(parents=True, exist_ok=True)
            with open(embs_file, 'wb') as handle:
                pickle.dump(embs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('[%s] Saved embeddings (%d) to %s', self.dataset, len(embs_dict), embs_file)
            try:
                del embs
                del embs_dict
            except Exception as exc:
                logger.debug('[%s] Failed to delete embeddings from memory: %s', self.dataset, str(exc))
                print(f"{self.dataset} Warning: failed to delete embeddings from memory: {str(exc)}")
            try:
                del model
            except Exception as exc:
                logger.debug('[%s] Failed to delete model from memory: %s', self.dataset, str(exc))
                print(f"{self.dataset} Warning: failed to delete model from memory: {str(exc)}")
            gc.collect()
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception:
            logger.exception('[%s] Failed during embedding computation', self.dataset)
            return False

    @staticmethod
    def _get_log_files(logs_dir: str) -> List[str]:
        """
        Get list of log files from a directory.

        Filters out hidden files (starting with '.'), pickle files, and
        already-structured files.

        Parameters
        ----------
        logs_dir : str
            Directory containing log files.

        Returns
        -------
        list[str]
            Sorted list of log file names (not full paths).
        """
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            return []
        files = []
        for item in logs_path.iterdir():
            if not item.is_file():
                continue
            name = item.name
            if name.startswith('.'):
                continue
            if name.endswith(constants.EXT_PICKLE) or name.endswith(constants.EXT_PKL) or name.endswith(constants.EXT_PICKLE_ALT):
                continue
            if name.endswith(constants.EXT_STRUCTURED_CSV) or name.endswith(constants.EXT_STRUCTURED_PICKLE):
                continue
            files.append(name)
        return sorted(files)

    def _parse_with_drain(self, args) -> bool:
        """
        Parse logs using the Drain parser and compute embeddings.

        Reads raw log files, applies Drain parser to extract structured content,
        collects unique messages, and computes embeddings using SentenceTransformer.

        Parameters
        ----------
        args : object
            Arguments object containing dataset, sequence_type, window_size, model, 
            device, and dry_run attributes for display purposes.

        Returns
        -------
        bool
            True if parsing and embedding computation succeeded, False otherwise.
        """
        try:
            from utils.logparsers import Drain
        except Exception:
            import sys as _sys
            script_dir = str(Path(__file__).parent.parent)
            if script_dir not in _sys.path:
                _sys.path.insert(0, script_dir)
            from utils.logparsers import Drain

        if not InputValidator.validate_settings_dict(self.settings, ['in_dir', 'out_dir', 'embs_dir', 'log_format', 'depth', 'st', 'regex'], self.dataset):
            return False

        logs_dir = Path(self.dataset_dir) / self.settings['in_dir']
        output_dir = Path(self.dataset_dir) / self.settings['out_dir']
        
        # Validate input directory exists
        try:
            InputValidator.validate_directory_exists(str(logs_dir), f"Logs directory for {self.dataset}")
        except FileNotFoundError as e:
            logger.error('[%s] %s', self.dataset, str(e))
            print(str(e))
            return False
        
        if self.dry_run:
            print("[STEP 1/2] [DRY RUN] Preprocessing with Drain parser")
            print(f"[DRY RUN] Would parse logs from: {logs_dir}")
            print(f"[DRY RUN] Would save structured logs to: {output_dir}")
            log_files = self._get_log_files(str(logs_dir))
            print(f"[DRY RUN] Found {len(log_files)} log file(s) to parse: {', '.join(log_files[:5])}{'...' if len(log_files) > 5 else ''}")
            return True
        
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = Drain.LogParser(log_format=self.settings['log_format'], indir=str(logs_dir), outdir=str(output_dir),
                                 depth=self.settings['depth'], st=self.settings['st'], rex=self.settings['regex'], keep_para=False)

        if not logs_dir.exists():
            logger.error('[%s] Logs directory not found: %s', self.dataset, logs_dir)
            print(f"Logs directory not found: {logs_dir}")
            return False

        all_messages = [constants.UNKNOWN_TOKEN, constants.NAN_TOKEN]
        log_files = self._get_log_files(str(logs_dir))

        print()
        print("="*70)
        print(f"CoLog Ground Truth Extraction")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Sequence Type: {args.sequence_type}")
        print(f"Window Size: {args.window_size}")
        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        if args.dry_run:
            print(f"Mode: DRY RUN (preview only, no changes will be made)")
        print("="*70)
        print()

        print(f"[STEP 1/2] Preprocessing dataset '{args.dataset}'...")
        print()

        print(f"[{self.dataset}] Found {len(log_files)} log file(s) to parse")
        for log_file in log_files:
            structured_csv = output_dir / (log_file + constants.EXT_STRUCTURED_CSV)
            if structured_csv.exists() and not self.force:
                try:
                    df = pd.read_csv(structured_csv)
                    msgs = list(df[constants.COL_CONTENT].unique())
                except Exception:
                    logger.exception('[%s] Failed to read structured CSV %s', self.dataset, structured_csv)
                    msgs = parser.parse(log_file)
                    logger.info('[%s] Re-parsed %s due to read failure', self.dataset, log_file)
                    print(f"Re-parsing {log_file} due to read failure.")
            else:
                try:
                    msgs = parser.parse(log_file)
                    logger.info('[%s] Parsed %s', self.dataset, log_file)
                    print(f"Parsed {log_file}.")
                except Exception:
                    logger.exception('[%s] Failed to parse %s', self.dataset, log_file)
                    continue
            all_messages.extend(msgs)

        try:
            del parser
        except Exception as e:
            logger.debug('[%s] Failed to delete parser object: %s', self.dataset, e)
        gc.collect()

        unique_msgs = list(dict.fromkeys(all_messages))
        embs_dir = Path(self.dataset_dir) / self.settings['embs_dir']
        embs_dir.mkdir(parents=True, exist_ok=True)
        embs_file = embs_dir / (self.dataset + constants.SUFFIX_EMBEDDINGS)
        return self._compute_and_save_embeddings(unique_msgs, str(embs_file))

    def _parse_with_ner(self, args) -> bool:
        """
        Parse logs using the NER parser and compute embeddings.

        Reads raw log files, applies NER parser to extract structured entities,
        collects unique messages, and computes embeddings using SentenceTransformer.

        Parameters
        ----------
        args : object
            Arguments object containing dataset, sequence_type, window_size, model, 
            device, and dry_run attributes for display purposes.

        Returns
        -------
        bool
            True if parsing and embedding computation succeeded, False otherwise.
        """
        try:
            from utils.logparsers.nerlogparser.nerlogparser_v2.nerlogparser import Nerlogparser
        except Exception:
            import sys as _sys
            script_dir = str(Path(__file__).parent.parent)
            if script_dir not in _sys.path:
                _sys.path.insert(0, script_dir)
            from utils.logparsers.nerlogparser.nerlogparser_v2.nerlogparser import Nerlogparser

        if not InputValidator.validate_settings_dict(self.settings, ['in_dir', 'out_dir', 'embs_dir'], self.dataset):
            return False

        logs_dir = Path(self.dataset_dir) / self.settings['in_dir']
        output_dir = Path(self.dataset_dir) / self.settings['out_dir']
        
        # Validate input directory exists
        try:
            InputValidator.validate_directory_exists(str(logs_dir), f"Logs directory for {self.dataset}")
        except FileNotFoundError as e:
            logger.error('[%s] %s', self.dataset, str(e))
            print(str(e))
            return False
        
        if self.dry_run:
            print("[STEP 1/2] [DRY RUN] Preprocessing with NER parser")
            print(f"[DRY RUN] Would parse NER logs from: {logs_dir}")
            print(f"[DRY RUN] Would save structured logs to: {output_dir}")
            log_files = self._get_log_files(str(logs_dir))
            print(f"[DRY RUN] Found {len(log_files)} log file(s) to parse: {', '.join(log_files[:5])}{'...' if len(log_files) > 5 else ''}")
            return True
        
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = Nerlogparser()
        if not logs_dir.exists():
            logger.error('[%s] Logs directory not found: %s', self.dataset, logs_dir)
            print(f"Logs directory not found: {logs_dir}")
            return False

        log_files = self._get_log_files(str(logs_dir))
        all_messages = [constants.UNKNOWN_TOKEN, constants.NAN_TOKEN]

        print()
        print("="*70)
        print(f"CoLog Ground Truth Extraction")
        print("="*70)
        print(f"Dataset: {args.dataset}")
        print(f"Sequence Type: {args.sequence_type}")
        print(f"Window Size: {args.window_size}")
        print(f"Model: {args.model}")
        print(f"Device: {args.device}")
        if args.dry_run:
            print(f"Mode: DRY RUN (preview only, no changes will be made)")
        print("="*70)
        print()

        print(f"[STEP 1/2] Preprocessing dataset '{args.dataset}'...")
        print()

        print(f"[{self.dataset}] Found {len(log_files)} log file(s) to parse")

        for log_file in log_files:
            output_file = output_dir / (log_file + constants.EXT_STRUCTURED_PICKLE)
            if output_file.exists() and not self.force:
                try:
                    with open(str(output_file), 'rb') as handle:
                        parsed_logs = pickle.load(handle)
                except Exception:
                    logger.exception('[%s] Failed to load structured parse %s', self.dataset, output_file)
                    parsed_logs = parser.parse_logs(str(logs_dir / log_file))
                    logger.info('[%s] Re-parsed %s due to load failure', self.dataset, log_file)
                    print(f"Re-parsing {log_file} due to load failure.")
            else:
                parsed_logs = parser.parse_logs(str(logs_dir / log_file))
                logger.info('[%s] Parsed %s', self.dataset, log_file)
                print(f"Parsed {log_file}.")

            log_messages = []
            for _, entities in parsed_logs.items():
                if entities.get('message', '') != '' and entities['message'] not in log_messages:
                    log_messages.append(entities['message'])
            all_messages.extend(log_messages)

            if not (output_file.exists() and not self.force):
                with open(str(output_file), 'wb') as handle:
                    pickle.dump(parsed_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

            try:
                del parsed_logs
                del log_messages
            except Exception as exc:
                logger.debug('[%s] Failed to delete parsed logs from memory: %s', self.dataset, str(exc))
                print(f"{self.dataset} Warning: failed to delete parsed logs from memory: {str(exc)}")
            gc.collect()

        try:
            del parser
        except Exception as exc:
            logger.debug('[%s] Failed to delete NER parser from memory: %s', self.dataset, str(exc))
            print(f"{self.dataset} Warning: failed to delete NER parser from memory: {str(exc)}")
        gc.collect()

        unique_msgs = list(dict.fromkeys(all_messages))
        embs_dir = Path(self.dataset_dir) / self.settings['embs_dir']
        embs_dir.mkdir(parents=True, exist_ok=True)
        embs_file = embs_dir / (self.dataset + constants.SUFFIX_EMBEDDINGS)
        return self._compute_and_save_embeddings(unique_msgs, str(embs_file))

    def run(self, args) -> bool:
        """
        Run the preprocessing pipeline for the dataset.

        Dispatches to the appropriate parser (Drain or NER) based on dataset type,
        or skips processing if dataset is not in known groups.

        Parameters
        ----------
        args : object
            Arguments object containing dataset, sequence_type, window_size, model, 
            device, and dry_run attributes for display purposes.

        Returns
        -------
        bool
            True if preprocessing succeeded or was skipped (dataset not in known
            groups), False on failure.
        """
        try:
            if not self.settings:
                logger.error('[%s] No settings found for preprocess', self.dataset)
                return False
            if self.dataset in constants.LOGS_DRAIN:
                logger.info('[%s] Dataset in DRAIN preprocess group: processing', self.dataset)
                # print(f"Dataset {self.dataset} in DRAIN preprocess group: processing")
                return self._parse_with_drain(args)
            elif self.dataset in constants.LOGS_NER:
                logger.info('[%s] Dataset in NER preprocess group: processing', self.dataset)
                # print(f"Dataset {self.dataset} in NER preprocess group: processing")
                return self._parse_with_ner(args)
            else:
                logger.info('[%s] Dataset not in known preprocess groups: skipping', self.dataset)
                return True
        except Exception:
            logger.exception('[%s] Preprocessing failed', self.dataset)
            return False
