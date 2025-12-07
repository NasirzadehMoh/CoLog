"""
extraction/groundtruth_type1.py — Type1 GroundTruth extraction

This module contains the Type1GroundTruthExtractor class for datasets where
labels are available as a 'Level' column in the parsed CSV (e.g., 'WARN' -> anomaly).

Applies to datasets: hadoop, zookeeper
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


class Type1GroundTruthExtractor(GroundTruthExtractor):
    """
    Ground truth extraction for Type1 datasets (hadoop, zookeeper).
    
    These datasets have a 'Level' column in the parsed CSV where 'WARN'
    indicates anomalies and other levels indicate normal messages.
    """

    def extract(self) -> None:
        """
        Build ground truth for datasets with Level column.
        
        This method expects structured CSV files with columns 'Level' and
        'Content'. The label assignment maps 'WARN' to 0 (anomaly) and
        everything else to 1 (normal).

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
        
        parsed_logs = [f.name for f in logs_dir.iterdir() if f.is_file()]
        logger.info('[%s] Type1 extraction start. files=%d window=%d sequence_type=%s',
                self.dataset, len(parsed_logs), self.window_size, self.sequence_type)
        print(f"[{self.dataset}] Starting Type1 extraction for {len(parsed_logs)} file(s)...")

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

            # Read CSV with only needed columns to minimize memory usage
            try:
                parsed_df = pd.read_csv(str(parsed_file_path), usecols=[constants.COL_LEVEL, constants.COL_CONTENT])
                # Validate required columns
                InputValidator.validate_dataframe_columns(parsed_df, [constants.COL_LEVEL, constants.COL_CONTENT], self.dataset)
            except DataFormatError as e:
                logger.error('[%s] %s in file %s', self.dataset, str(e), parsed_file)
                print(f"  [X] {str(e)}")
                continue
            except Exception as e:
                logger.error('[%s] Failed to read CSV %s: %s', self.dataset, parsed_file_path, str(e))
                print(f"  [X] Failed to read {parsed_file}: {str(e)}")
                continue
            
            messages_list = parsed_df[constants.COL_CONTENT].tolist()
            labels_list = parsed_df[constants.COL_LEVEL].tolist()
            del parsed_df
            gc.collect()

            file_hash = hashlib.md5(parsed_file.split('.')[0].encode('utf-8')).hexdigest()[0:8]

            for index, row in enumerate(messages_list):
                index_list, sequence_list = self._build_sequence(messages_list, index, self.window_size, self.sequence_type)
                cleaned_message = self._clean_message(row)
                message_label = 0 if labels_list[index] == constants.LABEL_WARN else 1

                if not self.groundbreaking:
                    label = message_label
                else:
                    sequence_label = 0 if constants.LABEL_WARN in [labels_list[i] for i in index_list] else 1
                    label = self._compute_groundbreaking_label(message_label, sequence_label)

                message_id = file_hash + str([index])
                messages[message_id] = np.array(cleaned_message, dtype=constants.NUMPY_STRING_DTYPE)
                sequences[message_id] = sequence_list
                labels[message_id] = int(label)
                keys.append(message_id)
            
            # Clean up memory after processing each file
            del messages_list, labels_list
            gc.collect()
            # print(f"    [OK] Completed {parsed_file.split('.')[0]}")

        self._save_all_groundtruth_files(messages, sequences, labels, keys)
        self._split_and_save_train_valid_test(messages, sequences, labels, 'Type1')
