"""
cli.py — Command-line interface for CoLog ground truth extraction

This module handles all argument parsing and validation for the ground truth
extraction script. It provides a clean separation between CLI concerns and
business logic.
"""

import argparse
from . import constants


def parse_arguments() -> argparse.Namespace:
    """
    Parse the CLI arguments for the ground truth generation script.

    The function returns a namespace with fields (defaults shown in parens):
        - dataset (hadoop): dataset name to process (e.g., 'hadoop', 'spark').
        - dataset_dir (datasets/): root path to datasets directory.
        - model (all-MiniLM-L6-v2): SentenceTransformer model used for embedding computation.
        - batch_size (64): Batch size for embedding computation.
        - device (auto): Device for embedding computation ('cpu', 'cuda', or auto-detect).
        - sequence_type (context): Sequence type to use ('background' or 'context').
        - window_size (1): Number of messages on each side of the current message.
        - train_ratio (0.6): Fraction of the (non-test) dataset to use for training.
        - valid_ratio (0.2): Fraction of the (non-test) dataset to use for validation.
        - force (False): If set, forces preprocessing and groundtruth regeneration even if files exist.
        - verbose (False): If set, enables DEBUG level logging.
        - groundbreaking (False): If set, enable multi-label/sequence-level labeling.
        - resample (False): If set, automatically run resampling after groundtruth extraction.
        - resample_method (None): Optional resampling method (e.g., 'TomekLinks', 'SMOTE').
        - random_seed (100): Random seed for reproducibility in dataset splitting and resampling.
        - dry_run (False): Preview what would be done without actually processing or writing files.

    Returns
    -------
    argparse.Namespace
        Namespace populated with parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
         prog='Script of CoLog: groundtruth.py',
         description="Extract log's groundtruth with CoLog.",
         epilog="Welcome to CoLog's world."
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        default='hadoop',
        choices=constants.LOGS_LIST,
        help="Dataset to extract groundtruth for. Choices: %(choices)s"
    )
    parser.add_argument(
        '--dataset-dir', dest='dataset_dir', type=str, default='datasets/',
        help='Path to the root datasets directory (default: datasets/)'
    )
    parser.add_argument(
        '--model', dest='model', type=str, default='all-MiniLM-L6-v2',
        help='SentenceTransformer model to use prior to ground truth.'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size', type=int, default=64,
        help='Batch size used during preprocessing embeddings.'
    )
    parser.add_argument(
        '--device', dest='device', type=str, default=constants.DEVICE_AUTO, choices=[constants.DEVICE_AUTO, constants.DEVICE_CPU, constants.DEVICE_CUDA],
        help="Device for preprocessing embeddings: 'cpu', 'cuda' or 'auto' (default: auto)"
    )
    parser.add_argument(
        '--sequence-type',
        dest='sequence_type',
        type=str,
        default=constants.SEQ_TYPE_CONTEXT,
        choices=[constants.SEQ_TYPE_BACKGROUND, constants.SEQ_TYPE_CONTEXT],
        help="Sequence type, 'background' or 'context' (default: context)"
    )
    parser.add_argument(
        '--window-size',
        dest='window_size',
        type=int,
        default=1,
        help="Number of messages on each side of the current message used in sequences. [e.g. --window-size 1]"
    )
    parser.add_argument(
        '--train-ratio',
        dest='train_ratio',
        type=float,
        default=constants.DEFAULT_TRAIN_RATIO,
        help=f"Fraction of the (non-test) dataset to use for training. [e.g. --train-ratio {constants.DEFAULT_TRAIN_RATIO}]"
    )
    parser.add_argument(
        '--valid-ratio',
        dest='valid_ratio',
        type=float,
        default=constants.DEFAULT_VALID_RATIO,
        help=f"Fraction of the (non-test) dataset to use for validation. [e.g. --valid-ratio {constants.DEFAULT_VALID_RATIO}]"
    )
    parser.add_argument(
        '--force', dest='force', action='store_true',
        help='Force re-preprocessing, re-extraction of groundtruth and resampling even if precomputed files exist.'
    )
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true',
        help='Enable verbose logging (debug level).'
    )
    parser.add_argument(
        '--groundbreaking', dest='groundbreaking', action='store_true',
        help='Enable groundbreaking ground truth extraction.'
    )

    # Resampling (class imbalance) options: optional. If provided, the
    # solver will run after groundtruth extraction using existing
    # embeddings (avoids recomputing embeddings).
    parser.add_argument(
        '--resample', dest='resample', action='store_true',
        help='Automatically run resampling after groundtruth extraction (requires --resample-method).'
    )
    parser.add_argument(
        '--resample-method', dest='resample_method', type=str, default=None,
        help='Optional: resampling method to run after groundtruth extraction.'
    )
    parser.add_argument(
        '--random-seed', dest='random_seed', type=int, default=constants.DEFAULT_RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {constants.DEFAULT_RANDOM_SEED})'
    )
    parser.add_argument(
        '--dry-run', dest='dry_run', action='store_true',
        help='Preview what would be done without actually processing or writing files.'
    )

    args = parser.parse_args()
    
    # Validate argument ranges
    if args.window_size < 0:
        parser.error(f"--window-size must be non-negative, got {args.window_size}")
    if not (0.0 < args.train_ratio < 1.0):
        parser.error(f"--train-ratio must be between 0 and 1, got {args.train_ratio}")
    if not (0.0 < args.valid_ratio < 1.0):
        parser.error(f"--valid-ratio must be between 0 and 1, got {args.valid_ratio}")
    if args.train_ratio + args.valid_ratio >= 1.0:
        parser.error(f"Sum of --train-ratio ({args.train_ratio}) and --valid-ratio ({args.valid_ratio}) must be less than 1.0")
    if args.batch_size <= 0:
        parser.error(f"--batch-size must be positive, got {args.batch_size}")
    
    return args
