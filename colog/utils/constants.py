"""
constants.py — Constants used across the CoLog training system

This module contains all constant values, configuration defaults, and magic strings
used throughout the CoLog training pipeline.
"""

import os
from typing import List

# ============================================================================
# Model Default Values
# ============================================================================

DEFAULT_EMBED_SIZE = 300
DEFAULT_SEQUENCES_FSIZE = 384
DEFAULT_LAYERS = 2
DEFAULT_HEADS = 4
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT_RATE = 0.1
DEFAULT_PROJECTION_SIZE = 2048

# ============================================================================
# Data Default Values
# ============================================================================

DEFAULT_LEN_MESSAGES = 60
DEFAULT_LEN_SEQUENCES = 60

# ============================================================================
# Training Default Values
# ============================================================================

DEFAULT_OUTPUT = 'runs/'
DEFAULT_NAME = 'hadoop'
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCH = 99
DEFAULT_EARLY_STOP = 3
DEFAULT_OPTIMIZER = 'Adam'
DEFAULT_OPTIMIZER_PARAMS = "{'betas': '(0.9, 0.98)', 'eps': '1e-9'}"
DEFAULT_LEARNING_RATE = 0.00005
DEFAULT_LR_DECAY = 0.5
DEFAULT_DECAY_TIMES = 3
DEFAULT_GRAD_CLIP = -1
DEFAULT_EVALUATION_START = 0
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_DEVICE = 'auto'

# ============================================================================
# Dataset Default Values
# ============================================================================

DEFAULT_DATASET_NAME = 'hadoop'
DEFAULT_SEQUENCE_TYPE = 'context'
DEFAULT_WINDOW_SIZE = 1

# ============================================================================
# Class Imbalance Default Values
# ============================================================================

DEFAULT_METHOD = 'TomekLinks'

# ============================================================================
# Tuning Default Values
# ============================================================================

DEFAULT_TUNER_SAMPLES = 4

# ============================================================================
# Dataset Lists
# ============================================================================

LOGS_LIST: List[str] = [
    'hadoop', 'spark', 'zookeeper', 'bgl', 'windows',
    'casper-rw', 'dfrws-2009-jhuisi', 'dfrws-2009-nssal',
    'honeynet-challenge7', 'honeynet-challenge5'
]

# ============================================================================
# Sequence Types
# ============================================================================

SEQUENCE_TYPES: List[str] = ['background', 'context']

# ============================================================================
# Resampling Methods
# ============================================================================

METHODS_LIST: List[str] = [
    'NeighbourhoodCleaningRule', 'OneSidedSelection', 'RandomUnderSampler',
    'TomekLinks', 'NearMiss', 'CondensedNearestNeighbour', 'EditedNearestNeighbours',
    'RepeatedEditedNearestNeighbours', 'AllKNN', 'InstanceHardnessThreshold'
]

# ============================================================================
# Device Types
# ============================================================================

DEVICE_AUTO = 'auto'
DEVICE_CPU = 'cpu'
DEVICE_CUDA = 'cuda'

# ============================================================================
# Directory Names
# ============================================================================

DIR_RUNS = 'runs/'
DIR_LOGS = 'runs/logs/'
DIR_CHECKPOINTS = 'checkpoints/'
DIR_RESULTS = 'results/'
DIR_MODELS = 'models/'
DIR_TUNED_MODEL = 'tuned_model/'
DIR_TUNED_RESULTS = 'tuned_results/'

# ============================================================================
# File Extensions
# ============================================================================

EXT_PICKLE = '.p'
EXT_PKL = '.pkl'
EXT_CSV = '.csv'
EXT_LOG = '.log'
EXT_PT = '.pt'
EXT_PTH = '.pth'

# ============================================================================
# File Name Suffixes
# ============================================================================

SUFFIX_BEST_CONFIG = '_best_config.p'
SUFFIX_TUNED = '_tuned.csv'
SUFFIX_CHECKPOINT = '_checkpoint.pt'
SUFFIX_BEST_MODEL = '_best_model.pt'
SUFFIX_FINAL_MODEL = '_final_model.pt'

# ============================================================================
# Log File Names
# ============================================================================

FILE_COLOG_LOG = 'colog_execution.log'

# ============================================================================
# Testing/Evaluation Default Values
# ============================================================================

DEFAULT_EVAL_SETS: List[str] = ['valid_set', 'test_set']
DEFAULT_NUM_CKPTS = 1
DEFAULT_DATASET = 'hadoop'

# ============================================================================
# Evaluation Set Options
# ============================================================================

EVAL_SET_OPTIONS = ['valid_set', 'test_set']

# ============================================================================
# Safe CPU count with fallback
# ============================================================================

CPU_COUNT = os.cpu_count() or 1