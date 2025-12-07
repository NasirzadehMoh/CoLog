"""
cli.py — Command-line interface for CoLog training and testing

This module handles all argument parsing and validation for both the CoLog training
and testing scripts. It provides a clean separation between CLI concerns and business logic.

The CLI supports multiple modes:
    1. Training mode: Train CoLog with specified hyperparameters
    2. Tuning mode: Perform hyperparameter search using Ray Tune
    3. Testing/Evaluation mode: Evaluate trained CoLog models
    4. Generalizability testing: Cross-dataset evaluation

Key features:
    - Comprehensive argument validation with meaningful error messages
    - Automatic device detection (CPU/CUDA/auto)
    - Output directory structure creation
    - Support for hyperparameter tuning with Ray Tune
    - Integration with class imbalance handling methods
    - Checkpoint ensembling for improved predictions
    - Full report generation with confusion matrix, ROC, PR curves

Usage examples:
    # Training
    python train.py --dataset hadoop --batch-size 32 --max-epoch 50
    
    # Testing/Evaluation
    python test.py --checkpoints-path runs/model_path/model/
    
    # Ensemble evaluation
    python test.py --checkpoints-path runs/model_path/model/ --ensemble True --num-ckpts 5
    
    # Generalizability testing
    python test.py --eval-generalizability True --name spark --dataset spark context_1

Dependencies: argparse, torch, pathlib, ray[tune]
"""

import argparse
import sys
import random
import logging
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any

import torch
from ray import tune

from . import constants

# Module-level logger for CLI operations
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent 'No handler could be found' warnings if imported before logging configured
    logger.addHandler(logging.NullHandler())


def parse_train_arguments() -> Union[argparse.Namespace, Tuple[Dict[str, Any], argparse.Namespace]]:
    """
    Parse and validate CLI arguments for the CoLog training script.

    This function handles comprehensive argument parsing, validation, and
    processing including device selection, output path creation, and tuning
    configuration setup. It provides extensive validation with clear error
    messages to help users quickly identify and fix configuration issues.

    The function returns a namespace with the following field groups:

    Model Architecture Arguments:
        - embedding_size (int): Length of each word's semantic vector; input
            size for message adapter (LSTM layer). Default: 300.
        - sequences_fsize (int): Dimension of sequence features; input size
            for sequence adapter (Linear layer). Default: 384.
        - layers (int): Number of collaborative transformer layers. Default: 2.
        - heads (int): Number of attention heads in collaborative transformer.
            Default: 4.
        - hidden_size (int): Size of collaborative transformer hidden layers.
            Default: 256.
        - dropout_rate (float): Dropout probability for regularization; rate
            of nodes randomly excluded from each update cycle. Default: 0.1.
        - projection_size (int): Size of projection layer on top of
            collaborative transformers for modality fusion. Default: 2048.

    Data Processing Arguments:
        - len_messages (int): Maximum length of log messages (tokens).
            Default: 60.
        - len_sequences (int): Maximum length of log sequences. Default: 60.

    Training Configuration Arguments:
        - output (str): Path to output results directory. Default: 'runs/'.
        - name (str): Name identifier for the CoLog model. Default: 'hadoop'.
        - batch_size (int): Number of samples processed before model update.
            Default: 32.
        - max_epoch (int): Maximum number of training epochs. Default: 99.
        - early_stop (int): Early stopping patience (epochs without improvement).
            Default: 3.
        - optimizer (str): Optimizer function name. Default: 'Adam'.
        - optimizer_params (dict): Optimizer parameters as dictionary.
            Default: {'betas': (0.9, 0.98), 'eps': 1e-9}.
        - learning_rate (float): Initial learning rate. Default: 0.00005.
        - lr_decay (float): Learning rate decay coefficient. Default: 0.5.
        - decay_times (int): Number of learning rate reductions. Default: 3.
        - grad_norm_clip (float): Gradient clipping max norm; -1 disables.
            Default: -1.
        - evaluation_start (int): Epoch to start evaluation after. Default: 0.
        - random_seed (int): Random seed for reproducibility. Default: random.
        - train_ratio (float): Fraction of dataset for training. Default: 0.6.
        - device (str): Device for training ('cpu', 'cuda', or 'auto').
            Default: 'auto'.

    Dataset Arguments:
        - dataset (str): Dataset name to train on. Default: 'hadoop'.
        - sequence_type (str): Sequence construction type ('background' or
            'context'). Default: 'context'.
        - window_size (int): Number of messages on each side of current
            message in sequences. Default: 1.

    Class Imbalance Arguments:
        - resample_method (str): Method for handling class imbalance.
            Default: 'TomekLinks'.

    Hyperparameter Tuning Arguments:
        - tuning (bool): Enable hyperparameter tuning mode. Default: False.
        - train_best_model (bool): Train with best config after tuning.
            Only valid when tuning=True. Default: False.
        - tuner_samples (int): Number of tuning configurations to sample.
            Default: 4.

    Processed Fields (added by this function):
        - loss_fn (torch.nn.Module): CrossEntropyLoss function for training
            (supports both binary and multiclass classification).
        - optimizer_params (dict): Parsed optimizer parameters dictionary.
        - output_path (str): Full output directory path with experiment name.
        - checkpoint_path (str): Model checkpoint directory path.
        - result_path (str): Results directory path for metrics and logs.
        
        Note: num_classes is dynamically detected from the dataset's ground
        truth labels during data loading, not set in arguments.

    Returns
    -------
    argparse.Namespace or tuple
        If tuning is disabled: Returns the augmented args namespace with all
            parsed and processed arguments.
        If tuning is enabled: Returns tuple of (config dict for Ray Tune,
            args namespace with string placeholders for tunable parameters).

    Raises
    ------
    SystemExit
        If any argument validation fails or if CUDA is requested but unavailable.

    Notes
    -----
    - Device selection: 'auto' automatically detects CUDA availability
    - Output directory structure: output/name__config_params/model and results
    - Tuning mode uses Ray Tune's grid_search and choice for hyperparameters
    - All argument ranges are validated with clear error messages

    Examples
    --------
    >>> args = parse_arguments()
    >>> print(args.dataset, args.batch_size, args.device)
    hadoop 32 cuda
    
    >>> config, args = parse_arguments()  # with --tuning flag
    >>> print(config['layers'])
    <ray.tune.search.sample.Categorical object>
    """
    parser = argparse.ArgumentParser(
        prog='CoLog Training Script',
        description='Train CoLog for log anomaly detection using collaborative transformers',
        epilog="Welcome to CoLog's world.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========================================================================
    # Model Architecture Arguments
    # ========================================================================
    parser.add_argument(
        '--embedding-size',
        dest='embedding_size',
        type=int,
        default=constants.DEFAULT_EMBED_SIZE,
        help=f'Dimension of word embeddings (input to message adapter LSTM). (default: {constants.DEFAULT_EMBED_SIZE})'
    )
    parser.add_argument(
        '--sequences-fsize',
        dest='sequences_fsize',
        type=int,
        default=constants.DEFAULT_SEQUENCES_FSIZE,
        help=f'Dimension of sequence features (input to sequence adapter). (default: {constants.DEFAULT_SEQUENCES_FSIZE})'
    )
    parser.add_argument(
        '--layers',
        dest='layers',
        type=int,
        default=constants.DEFAULT_LAYERS,
        help=f'Number of collaborative transformer layers. (default: {constants.DEFAULT_LAYERS})'
    )
    parser.add_argument(
        '--heads',
        dest='heads',
        type=int,
        default=constants.DEFAULT_HEADS,
        help=f'Number of attention heads in collaborative transformer. (default: {constants.DEFAULT_HEADS})'
    )
    parser.add_argument(
        '--hidden-size',
        dest='hidden_size',
        type=int,
        default=constants.DEFAULT_HIDDEN_SIZE,
        help=f'Size of collaborative transformer hidden layers. (default: {constants.DEFAULT_HIDDEN_SIZE})'
    )
    parser.add_argument(
        '--dropout-rate',
        dest='dropout_rate',
        type=float,
        default=constants.DEFAULT_DROPOUT_RATE,
        help=f'Dropout probability for regularization. (default: {constants.DEFAULT_DROPOUT_RATE})'
    )
    parser.add_argument(
        '--projection-size',
        dest='projection_size',
        type=int,
        default=constants.DEFAULT_PROJECTION_SIZE,
        help=f'Size of projection layer for modality fusion. (default: {constants.DEFAULT_PROJECTION_SIZE})'
    )

    # ========================================================================
    # Data Processing Arguments
    # ========================================================================
    parser.add_argument(
        '--len-messages',
        dest='len_messages',
        type=int,
        default=constants.DEFAULT_LEN_MESSAGES,
        help=f'Maximum length of log messages (tokens). (default: {constants.DEFAULT_LEN_MESSAGES})'
    )
    parser.add_argument(
        '--len-sequences',
        dest='len_sequences',
        type=int,
        default=constants.DEFAULT_LEN_SEQUENCES,
        help=f'Maximum length of log sequences. (default: {constants.DEFAULT_LEN_SEQUENCES})'
    )

    # ========================================================================
    # Training Configuration Arguments
    # ==============================================================================================================================
    # Training Configuration Arguments
    # ========================================================================
    parser.add_argument(
        '--output',
        dest='output',
        type=str,
        default=constants.DEFAULT_OUTPUT,
        help=f'Path to output results directory. (default: {constants.DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--name',
        dest='name',
        type=str,
        default=constants.DEFAULT_NAME,
        help=f'Name identifier for the CoLog model. (default: {constants.DEFAULT_NAME})'
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type=int,
        default=constants.DEFAULT_BATCH_SIZE,
        help=f'Number of samples processed before model update. (default: {constants.DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--max-epoch',
        dest='max_epoch',
        type=int,
        default=constants.DEFAULT_MAX_EPOCH,
        help=f'Maximum number of training epochs. (default: {constants.DEFAULT_MAX_EPOCH})'
    )
    parser.add_argument(
        '--early-stop',
        dest='early_stop',
        type=int,
        default=constants.DEFAULT_EARLY_STOP,
        help=f'Early stopping patience (epochs without improvement). (default: {constants.DEFAULT_EARLY_STOP})'
    )
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        type=str,
        default=constants.DEFAULT_OPTIMIZER,
        help=f'Optimizer function name. (default: {constants.DEFAULT_OPTIMIZER})'
    )
    parser.add_argument(
        '--optimizer-params',
        dest='optimizer_params',
        type=str,
        default=constants.DEFAULT_OPTIMIZER_PARAMS,
        help=f"Optimizer parameters as string dict. Example: {constants.DEFAULT_OPTIMIZER_PARAMS} (default: {constants.DEFAULT_OPTIMIZER_PARAMS})"
    )
    parser.add_argument(
        '--learning-rate',
        dest='learning_rate',
        type=float,
        default=constants.DEFAULT_LEARNING_RATE,
        help=f'Initial learning rate. (default: {constants.DEFAULT_LEARNING_RATE})'
    )
    parser.add_argument(
        '--lr-decay',
        dest='lr_decay',
        type=float,
        default=constants.DEFAULT_LR_DECAY,
        help=f'Learning rate decay coefficient. (default: {constants.DEFAULT_LR_DECAY})'
    )
    parser.add_argument(
        '--decay-times',
        dest='decay_times',
        type=int,
        default=constants.DEFAULT_DECAY_TIMES,
        help=f'Number of learning rate reductions. (default: {constants.DEFAULT_DECAY_TIMES})'
    )
    parser.add_argument(
        '--grad-clip',
        dest='grad_norm_clip',
        type=float,
        default=constants.DEFAULT_GRAD_CLIP,
        help=f'Gradient clipping max norm; -1 disables clipping. (default: {constants.DEFAULT_GRAD_CLIP})'
    )
    parser.add_argument(
        '--evaluation-start',
        dest='evaluation_start',
        type=int,
        default=constants.DEFAULT_EVALUATION_START,
        help=f'Epoch to start evaluation after. (default: {constants.DEFAULT_EVALUATION_START})'
    )
    parser.add_argument(
        '--random-seed',
        dest='random_seed',
        type=int,
        default=random.randint(0, 9999999),
        help='Random seed for reproducibility. (default: random)'
    )
    parser.add_argument(
        '--train-ratio',
        dest='train_ratio',
        type=float,
        default=None,
        help=f'Fraction of dataset for training. If not specified, auto-detects from available groundtruth. (default: auto-detect)'
    )
    parser.add_argument(
        '--device',
        dest='device',
        type=str,
        default=constants.DEFAULT_DEVICE,
        choices=[constants.DEVICE_AUTO, constants.DEVICE_CPU, constants.DEVICE_CUDA],
        help=f"Device for training: 'cpu', 'cuda', or 'auto' (default: {constants.DEFAULT_DEVICE})"
    )

    # ========================================================================
    # Dataset Arguments
    # ========================================================================
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        default=constants.DEFAULT_DATASET_NAME,
        choices=constants.LOGS_LIST,
        help=f'Dataset to train on. Choices: %(choices)s (default: {constants.DEFAULT_DATASET_NAME})'
    )
    parser.add_argument(
        '--sequence-type',
        dest='sequence_type',
        type=str,
        default=None,
        choices=constants.SEQUENCE_TYPES,
        help=f"Sequence construction type: 'background' or 'context'. If not specified, auto-detects from available groundtruth. (default: auto-detect)"
    )
    parser.add_argument(
        '--window-size',
        dest='window_size',
        type=int,
        default=None,
        help=f'Number of messages on each side of current message in sequences. If not specified, auto-detects from available groundtruth. (default: auto-detect)'
    )

    # ========================================================================
    # Class Imbalance Arguments
    # ========================================================================
    parser.add_argument(
        '--resample-method',
        dest='resample_method',
        type=str,
        default=None,
        choices=constants.METHODS_LIST,
        help=f'Method for handling class imbalance. Choices: %(choices)s (default: None - use original data)'
    )

    # ========================================================================
    # Hyperparameter Tuning Arguments
    # ================================================================================================================================
    # Hyperparameter Tuning Arguments
    # ========================================================================
    parser.add_argument(
        '--tuning',
        dest='tuning',
        action='store_true',
        help='Enable hyperparameter tuning mode using Ray Tune.'
    )
    parser.add_argument(
        '--train-bmodel',
        dest='train_best_model',
        action='store_true',
        help='Train with best configuration after tuning. Only valid when --tuning is set.'
    )
    parser.add_argument(
        '--tuner-samples',
        dest='tuner_samples',
        type=int,
        default=constants.DEFAULT_TUNER_SAMPLES,
        help=f'Number of tuning configurations to sample. (default: {constants.DEFAULT_TUNER_SAMPLES})'
    )

    args = parser.parse_args()

    # ========================================================================
    # Argument Validation
    # ========================================================================
    if args.embedding_size <= 0:
        parser.error(f"--embedding-size must be positive, got {args.embedding_size}")
    if args.sequences_fsize <= 0:
        parser.error(f"--sequences-fsize must be positive, got {args.sequences_fsize}")
    if args.layers <= 0:
        parser.error(f"--layers must be positive, got {args.layers}")
    if args.heads <= 0:
        parser.error(f"--heads must be positive, got {args.heads}")
    if args.hidden_size <= 0:
        parser.error(f"--hidden-size must be positive, got {args.hidden_size}")
    if not (0.0 <= args.dropout_rate < 1.0):
        parser.error(f"--dropout-rate must be between 0 and 1, got {args.dropout_rate}")
    if args.projection_size <= 0:
        parser.error(f"--projection-size must be positive, got {args.projection_size}")
    if args.len_messages <= 0:
        parser.error(f"--len-messages must be positive, got {args.len_messages}")
    if args.len_sequences <= 0:
        parser.error(f"--len-sequences must be positive, got {args.len_sequences}")
    if args.batch_size <= 0:
        parser.error(f"--batch-size must be positive, got {args.batch_size}")
    if args.max_epoch <= 0:
        parser.error(f"--max-epoch must be positive, got {args.max_epoch}")
    if args.early_stop < 0:
        parser.error(f"--early-stop must be non-negative, got {args.early_stop}")
    if args.learning_rate <= 0:
        parser.error(f"--learning-rate must be positive, got {args.learning_rate}")
    if not (0.0 < args.lr_decay <= 1.0):
        parser.error(f"--lr-decay must be between 0 and 1, got {args.lr_decay}")
    if args.decay_times < 0:
        parser.error(f"--decay-times must be non-negative, got {args.decay_times}")
    if args.evaluation_start < 0:
        parser.error(f"--evaluation-start must be non-negative, got {args.evaluation_start}")
    if args.window_size is not None and args.window_size < 0:
        parser.error(f"--window-size must be non-negative, got {args.window_size}")
    if args.train_ratio is not None and not (0.0 < args.train_ratio < 1.0):
        parser.error(f"--train-ratio must be between 0 and 1, got {args.train_ratio}")
    if args.tuner_samples <= 0:
        parser.error(f"--tuner-samples must be positive, got {args.tuner_samples}")
    if args.train_best_model and not args.tuning:
        parser.error("--train-bmodel requires --tuning to be set")

    # ========================================================================
    # Argument Processing
    # ========================================================================
    
    # Parse optimizer parameters from string to dict
    import ast
    try:
        args.optimizer_params = ast.literal_eval(args.optimizer_params)
        # Convert string tuples to actual tuples if needed (e.g., for betas)
        if 'betas' in args.optimizer_params and isinstance(args.optimizer_params['betas'], str):
            args.optimizer_params['betas'] = ast.literal_eval(args.optimizer_params['betas'])
        if 'eps' in args.optimizer_params and isinstance(args.optimizer_params['eps'], str):
            args.optimizer_params['eps'] = float(args.optimizer_params['eps'])
        logger.debug(f"Parsed optimizer_params: {args.optimizer_params}")
    except (ValueError, SyntaxError) as e:
        parser.error(f"Invalid optimizer-params format: {args.optimizer_params}. Error: {e}")
    
    # Set loss function for classification (supports both binary and multiclass)
    # Note: CrossEntropyLoss works for any number of classes (binary or multiclass)
    # The actual num_classes is automatically detected from the dataset's ground truth
    # labels during data loading (see GroundTruthLoader.num_classes attribute)
    args.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    # Set projection size if not specified
    if args.projection_size is None:
        args.projection_size = 4 * args.hidden_size
        logger.debug(f'Setting projection_size to 4 * hidden_size = {args.projection_size}')
    
    # ========================================================================
    # Auto-detect Groundtruth Configuration (if needed)
    # ========================================================================
    if args.sequence_type is None or args.window_size is None or args.train_ratio is None:
        try:
            from train.utils import detect_groundtruth_config
            
            logger.info(f"Auto-detecting groundtruth configuration for dataset '{args.dataset}'...")
            detected_config = detect_groundtruth_config(
                args.dataset,
                prefer_sequence_type=args.sequence_type,
                prefer_window_size=args.window_size,
                prefer_train_ratio=args.train_ratio,
                prefer_resample_method=args.resample_method
            )
            
            # Apply detected values only if not explicitly provided
            if args.sequence_type is None:
                args.sequence_type = detected_config['sequence_type']
                logger.info(f"Auto-detected sequence_type: {args.sequence_type}")
                print(f"Auto-detected sequence-type: {args.sequence_type}")
            
            if args.window_size is None:
                args.window_size = detected_config['window_size']
                logger.info(f"Auto-detected window_size: {args.window_size}")
                print(f"Auto-detected window-size: {args.window_size}")
            
            if args.train_ratio is None:
                args.train_ratio = detected_config['train_ratio'] if detected_config['train_ratio'] is not None else constants.DEFAULT_TRAIN_RATIO
                logger.info(f"Auto-detected train_ratio: {args.train_ratio}")
                print(f"Auto-detected train-ratio: {args.train_ratio}")

            if args.resample_method is None:
                args.resample_method = detected_config['resample_method']
                logger.info(f"Auto-detected resample_method: {args.resample_method}")
                print(f"Auto-detected resample-method: {args.resample_method}")
                
        except Exception as e:
            logger.error(f"Failed to auto-detect groundtruth configuration: {e}")
            print(f"Error: Failed to auto-detect groundtruth configuration: {e}")
            print(f"Please manually specify --sequence-type, --window-size, and --train-ratio")
            sys.exit(1)
    
    # Handle device selection and validation
    if args.device == constants.DEVICE_AUTO:
        if torch.cuda.is_available():
            args.device = constants.DEVICE_CUDA
            logger.info('Auto-detected CUDA device')
        else:
            logger.warning('CUDA not available, falling back to CPU')
            print('Warning: CUDA not available. Using CPU.')
            args.device = constants.DEVICE_CPU
    elif args.device == constants.DEVICE_CUDA and not torch.cuda.is_available():
        logger.error('CUDA device requested but not available')
        print('Error: CUDA device requested but not available.')
        sys.exit(1)
    
    # ========================================================================
    # Output Directory Creation
    # ========================================================================
    
    if not args.tuning:
        # Normal training mode: create structured output directories
        output_path = Path(args.output) / args.name
        args.output_path = str(output_path)
        
        # Create model checkpoint directory
        checkpoint_path = output_path / constants.DIR_MODELS
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        args.checkpoint_path = str(checkpoint_path)
        logger.info(f'Model checkpoints will be saved to: {checkpoint_path}')
        
        # Create results directory for metrics and logs
        result_path = output_path / constants.DIR_RESULTS
        result_path.mkdir(parents=True, exist_ok=True)
        args.result_path = str(result_path)
        logger.info(f'Results will be saved to: {result_path}')
        
        return args
    
    else:
        # Hyperparameter tuning mode: configure Ray Tune search space
        logger.info('Hyperparameter tuning mode enabled')
        
        # Define search space for hyperparameter optimization
        config = {
            'layers': tune.grid_search([2, 4, 6]),
            'heads': tune.grid_search([2, 4, 8]),
            'hidden_size': tune.grid_search([256, 512, 768]),
            'batch_size': tune.choice([32, 64, 96]),
            'dropout_rate': tune.choice([0.1, 0.2, 0.4]),
        }
        
        # Replace fixed values with config placeholders for tuning
        args.layers = "config['layers']"
        args.heads = "config['heads']"
        args.hidden_size = "int(config['hidden_size'])"
        args.batch_size = "int(config['batch_size'])"
        args.dropout_rate = "config['dropout_rate']"
        
        # Create tuning-specific output directories
        output_path = Path(args.output) / args.name
        args.output_path = str(output_path)
        
        checkpoint_path = output_path / constants.DIR_TUNED_MODEL
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        args.checkpoint_path = str(checkpoint_path)
        logger.info(f'Tuned model checkpoints will be saved to: {checkpoint_path}')
        
        result_path = output_path / constants.DIR_TUNED_RESULTS
        result_path.mkdir(parents=True, exist_ok=True)
        args.result_path = str(result_path)
        logger.info(f'Tuning results will be saved to: {result_path}')
        
        return config, args


def parse_test_arguments() -> argparse.Namespace:
    """
    Parse and validate CLI arguments for the CoLog testing/evaluation script.

    This function handles comprehensive argument parsing and validation for
    model evaluation, including support for checkpoint ensembling, cross-dataset
    generalization testing, and full metrics reporting.

    The function returns a namespace with the following field groups:

    Evaluation Configuration Arguments:
        - checkpoints_path (str): Path to checkpoints directory containing
            trained model files. Should contain files named 'best*.pt'.
        - eval_sets (List[str]): List of evaluation sets to use. Options are
            'valid_set' and 'test_set'.
        - ensemble (bool): Whether to ensemble predictions from multiple
            checkpoints. When True, predictions from num_ckpts checkpoints
            are averaged.
        - num_ckpts (int): Number of checkpoints to use for evaluation or
            ensembling. Checkpoints are selected from newest to oldest.

    Model Configuration Arguments:
        - name (str): Name identifier for the model. Used when evaluating
            generalizability on different datasets.
        - dataset (List[str]): Dataset specification as [name, sequence_type].
            The sequence_type should include window size (e.g., 'context_1').
            Used for generalizability evaluation.

    Generalizability Arguments:
        - eval_generalizability (bool): Whether to evaluate model on unseen
            datasets for generalization testing. When True, uses test_set only.

    Reporting Arguments:
        - plot_metrics (bool): Whether to generate full evaluation report with
            plots including confusion matrix, ROC curve, and PR curve.

    Returns
    -------
    argparse.Namespace
        Namespace containing all parsed and validated arguments.

    Raises
    ------
    SystemExit
        If any argument validation fails (e.g., invalid paths, invalid options).

    Notes
    -----
    - Checkpoints are automatically sorted by recency (newest first)
    - When ensemble=True and num_ckpts>1, predictions are averaged across checkpoints
    - For generalizability testing, only test_set is used regardless of eval_sets
    - Full report generates SVG and EPS files for all plots

    Examples
    --------
    >>> args = parse_test_arguments()
    >>> print(args.checkpoints_path, args.ensemble)
    """
    parser = argparse.ArgumentParser(
        prog='CoLog Testing/Evaluation Script',
        description='Evaluate trained CoLog models for log anomaly detection',
        epilog="Welcome to CoLog's evaluation world.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========================================================================
    # Evaluation Configuration Arguments
    # ========================================================================
    parser.add_argument(
        '--checkpoints-path',
        dest='checkpoints_path',
        type=str,
        default=None,
        help=f'Path to checkpoints directory containing trained model files. '
             f'If not provided, will be auto-generated from --dataset. '
             f'(default: auto-generated from dataset)'
    )
    parser.add_argument(
        '--eval-sets',
        dest='eval_sets',
        nargs='+',
        default=constants.DEFAULT_EVAL_SETS,
        choices=constants.EVAL_SET_OPTIONS,
        help=f'List of evaluation sets to use. Options: {constants.EVAL_SET_OPTIONS}. '
             f'(default: {constants.DEFAULT_EVAL_SETS})'
    )
    parser.add_argument(
        '--ensemble',
        dest='ensemble',
        action='store_true',
        help='Enable ensemble predictions from multiple checkpoints.'
    )
    parser.add_argument(
        '--num-ckpts',
        dest='num_ckpts',
        type=int,
        default=constants.DEFAULT_NUM_CKPTS,
        help=f'Number of checkpoints to use for evaluation/ensembling. '
             f'(default: {constants.DEFAULT_NUM_CKPTS})'
    )

    # ========================================================================
    # Model Configuration Arguments
    # ========================================================================
    parser.add_argument(
        '--name',
        dest='name',
        type=str,
        default=constants.DEFAULT_NAME,
        help=f'Name identifier for the model. Used for generalizability testing. '
             f'(default: {constants.DEFAULT_NAME})'
    )
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        default=constants.DEFAULT_DATASET,
        choices=constants.LOGS_LIST,
        help=f'Dataset name for evaluation. Choices: %(choices)s. '
             f'(default: {constants.DEFAULT_DATASET})'
    )

    # ========================================================================
    # Generalizability Arguments
    # ========================================================================
    parser.add_argument(
        '--eval-generalizability',
        dest='eval_generalizability',
        action='store_true',
        help='Enable model generalization evaluation on unseen datasets.'
    )

    # ========================================================================
    # Reporting Arguments
    # ========================================================================
    parser.add_argument(
        '--plot-metrics',
        dest='plot_metrics',
        action='store_true',
        help='Enable full evaluation report generation with metric plots.'
    )

    # Parse arguments
    args = parser.parse_args()

    # ========================================================================
    # Auto-generate checkpoints path if not provided
    # ========================================================================
    if args.checkpoints_path is None:
        # Generate checkpoints path based on dataset name
        args.checkpoints_path = str(Path(constants.DEFAULT_OUTPUT) / args.dataset / constants.DIR_MODELS)
        logger.info(f"Auto-generated checkpoints path from dataset: {args.checkpoints_path}")
        print(f"Auto-generated checkpoints-path: {args.checkpoints_path}")

    # ========================================================================
    # Argument Validation
    # ========================================================================
    
    # Validate checkpoints path exists
    checkpoints_path = Path(args.checkpoints_path)
    if not checkpoints_path.exists():
        logger.error(f"Checkpoints path does not exist: {args.checkpoints_path}")
        print(f"Error: Checkpoints path does not exist: {args.checkpoints_path}", file=sys.stderr)
        sys.exit(1)
    
    if not checkpoints_path.is_dir():
        logger.error(f"Checkpoints path is not a directory: {args.checkpoints_path}")
        print(f"Error: Checkpoints path must be a directory: {args.checkpoints_path}", file=sys.stderr)
        sys.exit(1)

    # Validate num_ckpts is positive
    if args.num_ckpts < 1:
        logger.error(f"num_ckpts must be at least 1, got: {args.num_ckpts}")
        print(f"Error: num_ckpts must be at least 1, got: {args.num_ckpts}", file=sys.stderr)
        sys.exit(1)

    # Validate dataset specification
    if len(args.dataset) < 2:
        logger.error(f"Dataset must include both name and sequence type, got: {args.dataset}")
        print(f"Error: Dataset must include both name and sequence type (e.g., hadoop context_1)", file=sys.stderr)
        sys.exit(1)

    # Validate dataset name
    dataset_name = args.dataset
    if dataset_name not in constants.LOGS_LIST:
        logger.warning(f"Dataset name '{dataset_name}' not in standard list: {constants.LOGS_LIST}")
        print(f"Warning: Dataset name '{dataset_name}' not in standard list. Proceeding anyway.", file=sys.stderr)

    # ========================================================================
    # Auto-generate result path based on checkpoints path
    # ========================================================================
    # For generalizability testing, save results under the test dataset directory
    # For standard evaluation, save results in the same parent directory as checkpoints
    if args.eval_generalizability:
        # e.g., if testing on dataset B, save to runs/dataset-B/results/
        output_base = Path(constants.DEFAULT_OUTPUT) / dataset_name
        result_path = output_base / constants.DIR_RESULTS
        logger.info(f"Generalizability mode: saving results to test dataset directory")
    else:
        # e.g., if checkpoints are in runs/casper-rw/models/, results go in runs/casper-rw/results/
        checkpoints_parent = checkpoints_path.parent
        result_path = checkpoints_parent / constants.DIR_RESULTS
    
    result_path.mkdir(parents=True, exist_ok=True)
    args.result_path = str(result_path)
    logger.info(f"Auto-generated result path: {args.result_path}")

    # Log configuration
    logger.info(f"Evaluation configuration:")
    logger.info(f"  Checkpoints path: {args.checkpoints_path}")
    logger.info(f"  Result path: {args.result_path}")
    logger.info(f"  Evaluation sets: {args.eval_sets}")
    logger.info(f"  Ensemble: {args.ensemble}")
    logger.info(f"  Number of checkpoints: {args.num_ckpts}")
    logger.info(f"  Model name: {args.name}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Eval generalizability: {args.eval_generalizability}")
    logger.info(f"  Plot metrics: {args.plot_metrics}")

    return args
