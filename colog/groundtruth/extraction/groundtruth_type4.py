"""
extraction/groundtruth_type4.py — Type4 GroundTruth extraction

This module contains the Type4GroundTruthExtractor class for NER-parsed datasets.

Input files are expected to be pickled structured dictionaries where each entry
contains a 'message' field. A wordlist is used to assign anomaly labels.

Applies to datasets: casper-rw, dfrws-2009-jhuisi, dfrws-2009-nssal,
honeynet-challenge7, honeynet-challenge5
"""

import pickle
from .groundtruth_base import GroundTruthExtractor
from utils import InputValidator
from utils import ValidationError
from utils import constants
from collections import OrderedDict
import hashlib
import numpy as np
from tqdm import tqdm
import logging
import gc


logger = logging.getLogger(__name__)


class Type4GroundTruthExtractor(GroundTruthExtractor):
    """
    Ground truth extraction for Type4 datasets (NER-parsed logs).
    
    These datasets are stored as pickle files containing dictionaries with
    'message' fields. Labels are assigned using wordlist-based heuristics.
    """

    def extract(self) -> None:
        """
        Build ground truth for NER-parsed datasets.

        Input files are pickled dictionaries. As in type2, a wordlist is used
        to assign anomaly labels. Most NER datasets produce train/valid/test
        splits; honeynet-challenge5 produces a generalization test set.

        Returns
        -------
        None
        """
        should_continue, logs_dir = self._validate_and_check_paths()
        if not should_continue:
            return

        parsed_logs = [f.name for f in logs_dir.iterdir() if f.is_file()]
        logger.info('[%s] Type4 extraction start. files=%d window=%d sequence_type=%s',
                self.dataset, len(parsed_logs), self.window_size, self.sequence_type)
        print(f"[{self.dataset}] Starting Type4 extraction for {len(parsed_logs)} file(s)...")

        messages = OrderedDict()
        sequences = OrderedDict()
        labels = OrderedDict()
        keys = []

        for parsed_file in tqdm(parsed_logs, total=len(parsed_logs), colour='#1b8079'):
            # print(f"  - Extracting {parsed_file.split('.')[0]} log's ground truth...")
            logger.debug('[%s] Processing file=%s', self.dataset, parsed_file)
            
            parsed_file_path = logs_dir / parsed_file
            log_type = parsed_file.split('.')[0].split('_')[0].lower()
            
            # Validate pickle file exists and can be loaded
            try:
                InputValidator.validate_pickle_file(parsed_file_path, expected_type=dict)
            except ValidationError as e:
                logger.error('[%s] %s', self.dataset, str(e))
                print(f"  [X] {str(e)}")
                continue

            # Read pickled NER-parsed log
            try:
                with open(parsed_file_path, 'rb') as handle:
                    parsed_log = pickle.load(handle)
            except Exception as e:
                logger.error('[%s] Failed to load pickle file %s: %s', self.dataset, parsed_file_path, str(e))
                print(f"  [X] Failed to load {parsed_file}: {str(e)}")
                continue

            messages_list = [parsed_log[i]['message'] for i in range(len(parsed_log))]
            wordlist = self._read_wordlist(log_type)
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
            del parsed_log, messages_list, wordlist
            gc.collect()
            # print(f"    [OK] Completed {parsed_file.split('.')[0]}")

        self._save_all_groundtruth_files(messages, sequences, labels, keys)

        # Most NER datasets get train/valid/test split, honeynet-challenge5 gets generalization set
        if self.dataset != 'honeynet-challenge5':
            self._split_and_save_train_valid_test(messages, sequences, labels, 'Type4')
        else:
            self._save_generalization_set(messages, sequences, labels, keys, 'Type4')
