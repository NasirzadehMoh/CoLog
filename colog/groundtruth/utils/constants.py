"""
constants.py — Constants used across the CoLog ground truth extraction system

This module contains all constant values, configuration defaults, and magic strings
used throughout the ground truth extraction and preprocessing pipeline.
"""

import os
from typing import List

# ============================================================================
# Magic Tokens
# ============================================================================

UNKNOWN_TOKEN = 'UNK'
NAN_TOKEN = 'NaN'

# ============================================================================
# Column Names
# ============================================================================

COL_LEVEL = 'Level'
COL_CONTENT = 'Content'
COL_LABEL = 'Label'

# ============================================================================
# File Extensions and Patterns
# ============================================================================

EXT_PICKLE = '.p'
EXT_PKL = '.pkl'
EXT_PICKLE_ALT = '.pickle'
EXT_STRUCTURED_CSV = '_structured.csv'
EXT_STRUCTURED_PICKLE = '_structured.p'
EXT_TXT = '.txt'
EXT_CSV = '.csv'
SUFFIX_EMBEDDINGS = '_embeddings.p'
SUFFIX_TRAIN_SET = '_train_set.p'
SUFFIX_VALID_SET = '_valid_set.p'
SUFFIX_TEST_SET = '_test_set.p'

# ============================================================================
# Directory Names
# ============================================================================

DIR_GROUNDTRUTH = 'groundtruth/'
DIR_WORDLIST = 'wordlists/'
DIR_LOG_EMBEDDINGS = 'log_embeddings/'
DIR_LOGS = 'runs/logs/'

# ============================================================================
# File Names
# ============================================================================

FILE_MESSAGES = 'messages'
FILE_SEQUENCES = 'sequences'
FILE_LABELS = 'labels'
FILE_KEYS = 'keys'
FILE_TRAIN_SET = 'train_set'
FILE_VALID_SET = 'valid_set'
FILE_TEST_SET = 'test_set'
FILE_GENERALIZATION = 'generalization'
FILE_COLOG_LOG = 'colog_execution.log'

# ============================================================================
# Special Label Values
# ============================================================================

LABEL_WARN = 'WARN'
LABEL_BGL_NORMAL = '-'

# ============================================================================
# Device Types
# ============================================================================

DEVICE_AUTO = 'auto'
DEVICE_CPU = 'cpu'
DEVICE_CUDA = 'cuda'

# ============================================================================
# Sequence Types
# ============================================================================

SEQ_TYPE_BACKGROUND = 'background'
SEQ_TYPE_CONTEXT = 'context'

# ============================================================================
# Subdirectory Patterns
# ============================================================================

SUBDIR_ALL = 'all/'
SUBDIR_RESAMPLED = 'resampled_groundtruth'

# ============================================================================
# Dataset Lists
# ============================================================================

LOGS_LIST: List[str] = [
    'hadoop', 'spark', 'zookeeper', 'bgl', 'windows',
    'casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal',
    'honeynet-challenge7', 'honeynet-challenge5', 'all'
]

LOGS_DRAIN: List[str] = ['hadoop', 'spark', 'zookeeper', 'bgl', 'windows']
LOGS_NER: List[str] = ['casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal', 'honeynet-challenge7', 'honeynet-challenge5']
LOGS_TYPE1: List[str] = ['hadoop', 'zookeeper']
LOGS_TYPE2: List[str] = ['spark', 'windows']
LOGS_TYPE3: List[str] = ['bgl']
LOGS_TYPE4: List[str] = LOGS_NER

# Datasets excluded from resampling (not used for training)
DATASETS_NO_RESAMPLING: List[str] = ['windows', 'honeynet-challenge5']

# ============================================================================
# Default Configuration Values
# ============================================================================

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2
DEFAULT_RANDOM_SEED = 100
NUMPY_STRING_DTYPE = '<U32'
EMBEDDING_DIMENSION = 384

# ============================================================================
# Resampling Methods
# ============================================================================

METHODS_LIST: List[str] = [
    'NeighbourhoodCleaningRule', 'OneSidedSelection', 'RandomUnderSampler',
    'TomekLinks', 'NearMiss', 'CondensedNearestNeighbour', 'EditedNearestNeighbours',
    'RepeatedEditedNearestNeighbours', 'AllKNN', 'InstanceHardnessThreshold'
]

# ============================================================================
# Safe CPU count with fallback
# ============================================================================

CPU_COUNT = os.cpu_count() or 1
