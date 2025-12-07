"""
extraction/groundtruth_type2.py — Type2 GroundTruth extraction

This module contains the Type2GroundTruthExtractor class for datasets where
labels are heuristically determined by searching a wordlist in the message content.

Applies to datasets: spark, windows
"""

from .groundtruth_base import GroundTruthExtractor
from utils import InputValidator
from utils import DataFormatError, FileNotFoundError
from utils import constants
import pandas as pd
from collections import OrderedDict
import hashlib
import numpy as np
from tqdm import tqdm
import logging
import gc


logger = logging.getLogger(__name__)


class Type2GroundTruthExtractor(GroundTruthExtractor):
    """
    Ground truth extraction for Type2 datasets (spark, windows).
    
    These datasets use wordlist-based labeling where the presence of specific
    keywords in the message content determines if it's an anomaly.
    """

    def extract(self) -> None:
        """
        Extract ground truth for datasets with wordlist-based labeling.

        The method reads only the 'Content' column from CSV; labels are
        assigned based on wordlist lookup. For Spark, the output includes
        train/valid/test splits. For Windows, produces a generalization test set.

        Returns
        -------
        None
        """
        should_continue, logs_dir = self._validate_and_check_paths()
        if not should_continue:
            return
        
        # Check if directory exists and has files
        if not logs_dir.exists():
            error_msg = f"Structured logs directory does not exist: {logs_dir}. Please run preprocessing first."
            logger.error('[%s] %s', self.dataset, error_msg)
            print(f"[X] {error_msg}")
            return
        
        log_type = self.dataset.lower()
        wordlist = self._read_wordlist(log_type)
        
        parsed_logs = [f.name for f in logs_dir.iterdir() if f.is_file()]
        logger.info('[%s] Type2 extraction start. files=%d window=%d sequence_type=%s',
                self.dataset, len(parsed_logs), self.window_size, self.sequence_type)
        print(f"[{self.dataset}] Starting Type2 extraction for {len(parsed_logs)} file(s)...")

        messages = OrderedDict()
        sequences = OrderedDict()
        labels = OrderedDict()
        keys = []

        for parsed_file in tqdm(parsed_logs, total=len(parsed_logs), colour='#1b8079'):
            # print(f"  - Extracting {parsed_file.split('.')[0]} log's ground truth...")
            logger.debug('[%s] Processing file=%s', self.dataset, parsed_file)
            parsed_file_path = logs_dir / parsed_file
            
            # Validate file exists
            try:
                InputValidator.validate_file_exists(parsed_file_path, f"Parsed log file")
            except FileNotFoundError as e:
                logger.error('[%s] %s', self.dataset, str(e))
                print(f"  [X] {str(e)}")
                continue

            # Read CSV with only needed column to minimize memory usage
            try:
                parsed_df = pd.read_csv(str(parsed_file_path), usecols=[constants.COL_CONTENT])
                # Validate required columns
                InputValidator.validate_dataframe_columns(parsed_df, [constants.COL_CONTENT], self.dataset)
            except DataFormatError as e:
                logger.error('[%s] %s in file %s', self.dataset, str(e), parsed_file)
                print(f"  [X] {str(e)}")
                continue
            except Exception as e:
                logger.error('[%s] Failed to read CSV %s: %s', self.dataset, parsed_file_path, str(e))
                print(f"  [X] Failed to read {parsed_file}: {str(e)}")
                continue
            
            messages_list = parsed_df[constants.COL_CONTENT].tolist()
            del parsed_df
            gc.collect()

            file_hash = hashlib.md5(parsed_file.split('.')[0].encode('utf-8')).hexdigest()[0:8]

            for index, row in enumerate(messages_list):
                _, sequence_list = self._build_sequence(messages_list, index, self.window_size, self.sequence_type)
                cleaned_message = self._clean_message(row)
                message_label = self._set_anomaly_label(wordlist, row)

                if not self.groundbreaking:
                    label = message_label
                else:
                    sequence_label = self._set_anomaly_label(wordlist, ' '.join(sequence_list))
                    label = self._compute_groundbreaking_label(message_label, sequence_label)

                message_id = file_hash + str([index])
                messages[message_id] = np.array(cleaned_message, dtype=constants.NUMPY_STRING_DTYPE)
                sequences[message_id] = sequence_list
                labels[message_id] = int(label)
                keys.append(message_id)

            # Clean up memory after processing each file
            del messages_list
            gc.collect()
            # print(f"    [OK] Completed {parsed_file.split('.')[0]}")

        self._save_all_groundtruth_files(messages, sequences, labels, keys)

        # Spark gets train/valid/test split, windows gets generalization set
        if self.dataset == 'spark':
            self._split_and_save_train_valid_test(messages, sequences, labels, 'Type2')
        else:
            self._save_generalization_set(messages, sequences, labels, keys, 'Type2')
