"""
constants.py — Constants used across the CoLog training system

This module contains all constant values, configuration defaults, and magic strings
used throughout the training and evaluation pipeline.
"""

from typing import List

# ============================================================================
# Magic Tokens
# ============================================================================

UNKNOWN_TOKEN = 'UNK'
NAN_TOKEN = 'NaN'

# ============================================================================
# File Extensions and Patterns
# ============================================================================

EXT_PICKLE = '.p'
EXT_PKL = '.pkl'
EXT_PICKLE_ALT = '.pickle'
EXT_NPY = '.npy'
EXT_LOG = '.log'

# ============================================================================
# File Name Patterns
# ============================================================================

FILE_TOKENS = 'tokens.pkl'
FILE_TRAIN_EMBEDDINGS = 'train_embeddings.npy'
FILE_TRAINING_LOG = 'training.log'
FILE_TRAINING_METRICS_LOG = 'training_metrics.log'

# ============================================================================
# Checkpoint Patterns
# ============================================================================

CHECKPOINT_PREFIX_BEST = 'best'
CHECKPOINT_TEMPLATE = 'best{seed}.pkl'
CHECKPOINT_TEMPLATE_WITH_ACC = 'best{accuracy:.2f}_{seed}.pkl'

# ============================================================================
# Directory Names
# ============================================================================

DIR_LOG_EMBEDDINGS = 'log_embeddings'
DIR_GROUNDTRUTH = 'groundtruth'
DIR_RESAMPLED = 'resampled_groundtruth'
DIR_MODEL = 'model'
DIR_MODELS = 'models'
DIR_RESULTS = 'results'
DIR_TRAIN_VALID_TEST_PREFIX = 'train_valid_test_'

# ============================================================================
# Data Split Names
# ============================================================================

SPLIT_TRAIN = 'train_set'
SPLIT_VALID = 'valid_set'
SPLIT_TEST = 'test_set'

# Valid data split names
VALID_SPLITS: List[str] = [SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST]

# ============================================================================
# Suffix Patterns
# ============================================================================

SUFFIX_EMBEDDINGS = '_embeddings.p'
SUFFIX_TRAIN_SET = '_train_set.p'
SUFFIX_VALID_SET = '_valid_set.p'
SUFFIX_TEST_SET = '_test_set.p'

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
# Ground Truth Data Attributes
# ============================================================================

ATTR_MESSAGES = 'messages'
ATTR_SEQUENCES = 'sequences'
ATTR_LABELS = 'labels'
ATTR_KEYS = 'keys'

# ============================================================================
# Default Configuration Values
# ============================================================================

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VALID_RATIO = 0.2
DEFAULT_RANDOM_SEED = 100
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_SPACY_EMBEDDING_DIM = 300

# ============================================================================
# NLTK Configuration
# ============================================================================

NLTK_LANGUAGE = 'english'
NLTK_STOPWORDS_PACKAGE = 'stopwords'

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

# Datasets with only train/test splits (no validation split)
UNSEEN_LOGS: List[str] = ['spark', 'windows', 'honeynet-challenge5']
# ============================================================================
# Model Configuration
# ============================================================================

# Default hyperparameters
DEFAULT_MAX_EPOCH = 100
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_HEADS = 4
DEFAULT_HIDDEN_SIZE = 256

# Training configuration
DEFAULT_EVAL_START = 5
DEFAULT_EARLY_STOP = 10
DEFAULT_LR_DECAY = 0.5
DEFAULT_DECAY_TIMES = 3
DEFAULT_GRAD_NORM_CLIP = 1.0

# ============================================================================
# Ray Tune Configuration
# ============================================================================

RAY_PERFECT_ACCURACY = 100.0
RAY_METRIC_ACCURACY = 'accuracy'

# ============================================================================
# System Paths
# ============================================================================

# Project root (used in Ray Tune)
PROJECT_ROOT = '/colog'

# Default dataset directory
DEFAULT_DATASET_DIR = 'datasets'
