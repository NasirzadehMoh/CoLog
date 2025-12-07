"""
train.py — Entry point for CoLog training and hyperparameter tuning

This script provides the main entry point for training the CoLog collaborative
transformer model for log anomaly detection. It supports two modes of operation:

1. Standard training: Train a model with fixed hyperparameters
2. Hyperparameter tuning: Use Ray Tune to search for optimal hyperparameters
   using ASHA (Asynchronous Successive Halving Algorithm) scheduling

The script handles:
    - Dataset loading and preprocessing via GroundTruthLoader
    - Model initialization with optional multi-GPU support (DataParallel)
    - Text embedding layer setup with pre-trained or learned embeddings
    - Training orchestration via the Trainer class
    - Hyperparameter search configuration and execution
    - Best configuration selection and model training
    - Results persistence (configurations, metrics, checkpoints)

Training workflow:
    1. Parse command-line arguments and configuration
    2. Set random seeds for reproducibility
    3. Load training and validation datasets
    4. Initialize model architecture and embedding layers
    5. Execute training loop with evaluation
    6. Save best model checkpoint and metrics

Tuning workflow:
    1. Define hyperparameter search space
    2. Configure ASHA scheduler for efficient search
    3. Run multiple training trials with different configurations
    4. Select best configuration based on validation accuracy
    5. Optionally train final model with best hyperparameters
    6. Save tuning results and best configuration

Usage examples:
    Standard training:
        python train.py --dataset hadoop --batch-size 32 --max-epoch 50 \\
            --learning-rate 0.001
    
    Hyperparameter tuning:
        python train.py --dataset spark --tuning --tuner-samples 20 \\
            --train-best-model
    
    Multi-GPU training:
        python train.py --dataset bgl --device cuda --batch-size 64

Files produced:
    - best<seed>.pkl: Best model checkpoint during training
    - <name>_best_config.p: Best hyperparameter configuration (tuning mode)
    - <name>_tuned.csv: Complete tuning results with all trials (tuning mode)
    - training_metrics.log: Detailed training history (via Trainer class)

Dependencies:
    torch, numpy, ray[tune], train.Trainer, train.utils.GroundTruthLoader,
    neuralnetwork.CollaborativeTransformer, neuralnetwork.TextEmbedding
"""

import logging
import os
import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Union, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from neuralnetwork import CollaborativeTransformer, TextEmbedding
from train import Trainer
from train import tune_model
from train.utils import GroundTruthLoader
from utils import parse_train_arguments
from utils import constants


# Ignore all warnings (comment out during debugging if needed)
warnings.filterwarnings("ignore", category=UserWarning)

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent 'No handler could be found' warnings if imported before logging configured
    logger.addHandler(logging.NullHandler())


def train_with_config(args) -> float:
    """
    Execute training pipeline with specified configuration.

    This function orchestrates the complete training workflow:
    1. Creates training and validation data loaders
    2. Initializes embedding layer and CoLog model
    3. Configures multi-GPU training if available
    4. Delegates training to Trainer class
    5. Returns final validation accuracy

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing all training parameters:
        - batch_size (int): Batch size for training and validation
        - device (torch.device): Device for model computation
        - Other model and training hyperparameters

    Returns
    -------
    float
        Final validation accuracy achieved during training.

    Notes
    -----
    - The validation dataset shares the token dictionary with training dataset
      to ensure consistent vocabulary mapping
    - DataLoader uses all available CPU cores for data loading
    - Pin memory is enabled for faster GPU transfer
    - Multi-GPU training uses DataParallel wrapper automatically
    - Model parameter count is printed for reference

    See Also
    --------
    Trainer.train_model : Core training loop implementation
    GroundTruthLoader : Dataset class for loading preprocessed logs
    """
    try:
        logger.info("Initializing datasets and data loaders")
        
        # Create datasets with shared vocabulary
        train_dataset = GroundTruthLoader('train_set', args)
        valid_dataset = GroundTruthLoader(
            'valid_set', 
            args, 
            train_dataset.tokens_dict
        )
        
        # Create data loaders with CPU-optimized settings
        # Use fewer workers for CPU to reduce overhead
        optimal_workers = min(4, constants.CPU_COUNT) if args.device == 'cpu' else constants.CPU_COUNT
        use_pin_memory = args.device == 'cuda'  # Only useful for GPU
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=optimal_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True if optimal_workers > 0 else False
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=optimal_workers,
            pin_memory=use_pin_memory,
            persistent_workers=True if optimal_workers > 0 else False
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        
        # Initialize embedding layer with vocabulary and pre-trained embeddings
        embedding_layer = TextEmbedding(
            args,
            train_dataset.vocab_size,
            train_dataset.embeddings
        )
        
        # Initialize CoLog collaborative transformer model
        # Use num_classes from dataset instead of args for flexibility
        model = CollaborativeTransformer(
            args, 
            train_dataset.vocab_size, 
            train_dataset.embeddings,
            num_classes=train_dataset.num_classes
        )
        
        # Configure multi-GPU training if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        # Move models to target device
        model.to(args.device)
        embedding_layer.to(args.device)
        
        # Calculate and log model parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Total CoLog parameters: {num_params:.2f}M")
        print(f"Total number of CoLog's parameters: {num_params:.2f} Million\n")
        
        # Execute training loop
        logger.info("Starting training pipeline")
        trainer = Trainer(args)
        final_accuracy = trainer.train_model(
            model,
            embedding_layer,
            train_loader,
            valid_loader
        )
        
        logger.info(f"Training completed with final accuracy: {final_accuracy:.4f}")
        return final_accuracy
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise



def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Configures random number generators for PyTorch, NumPy, and CUDA to ensure
    deterministic behavior. Also sets cuDNN flags for reproducible results.

    Parameters
    ----------
    seed : int
        Random seed value to use across all libraries.

    Notes
    -----
    - Setting cudnn.deterministic=True may reduce performance
    - cudnn.benchmark=False ensures deterministic algorithm selection
    - These settings are essential for reproducible research
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug(f"Random seeds set to {seed}")


def run_hyperparameter_tuning(config: dict, args) -> None:
    """
    Execute hyperparameter tuning using Ray Tune with ASHA scheduling.

    This function configures and runs a hyperparameter search using the
    Asynchronous Successive Halving Algorithm (ASHA) for efficient exploration.
    The best configuration is selected based on validation accuracy, and
    optionally a final model is trained with the optimal hyperparameters.

    Parameters
    ----------
    config : dict
        Hyperparameter search space configuration containing tunable parameters
        such as layers, heads, hidden_size, batch_size, dropout_rate.
    args : argparse.Namespace
        Configuration object with tuning settings:
        - tuner_samples (int): Number of trials to run
        - checkpoint_path (str): Directory for saving trial checkpoints
        - result_path (str): Directory for saving results
        - name (str): Experiment name for output files
        - train_best_model (bool): Whether to train final model with best config

    Returns
    -------
    None

    Notes
    -----
    - Uses ASHA scheduler with early stopping for efficient search
    - Trials are stopped early if they show poor performance
    - Results are saved to CSV with all trial metrics
    - Best configuration is saved as pickle file
    - If train_best_model=True, trains final model with optimal settings

    Files produced
    --------------
    - <name>_best_config.p: Best hyperparameter configuration
    - <name>_tuned.csv: All trial results sorted by accuracy

    See Also
    --------
    Trainer.tune_model : Single trial training function
    train_with_config : Final model training with best configuration
    """
    try:
        logger.info("Starting hyperparameter tuning with Ray Tune")
        
        # Configure ASHA scheduler for efficient hyperparameter search
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=4,
            grace_period=1,
            reduction_factor=2
        )
        
        # Configure progress reporter
        reporter = CLIReporter(
            metric_columns=['accuracy', 'training_iteration']
        )
        
        logger.info(f"Running {args.tuner_samples} tuning trials")
        

        
        # Execute hyperparameter search
        result = tune.run(
            partial(tune_model, checkpoint_dir=None, args=args),
            resources_per_trial={
                'cpu': constants.CPU_COUNT,
                'gpu': torch.cuda.device_count()
            },
            config=config,
            num_samples=args.tuner_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            storage_path=f'file:///colog/{args.checkpoint_path}'
        )
        
        # Extract best configuration
        best_config = result.get_best_config('accuracy', 'max')
        logger.info(f"Best configuration found: {best_config}")
        print(f"\n\nBest config: {best_config}\n\n")
        
        # Save best configuration
        best_config_path = Path(args.result_path) / f'{args.name}_best_config.p'
        with open(best_config_path, 'wb') as f:
            pickle.dump(best_config, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Best configuration saved to {best_config_path}")
        
        # Save all tuning results
        results_df_path = Path(args.result_path) / f'{args.name}_tuned.csv'
        tuning_results_df = result.results_df.sort_values(
            by=['accuracy', 'time_total_s'],
            ascending=[False, True]
        )
        tuning_results_df.to_csv(results_df_path)
        logger.info(f"Tuning results saved to {results_df_path}")
        
        # Train final model with best configuration
        if args.train_best_model:
            logger.info("Training final model with best hyperparameters")
            args.tuning = False
            args.layers = best_config['layers']
            args.heads = best_config['heads']
            args.hidden_size = best_config['hidden_size']
            args.batch_size = best_config['batch_size']
            args.dropout_rate = best_config['dropout_rate']
            
            final_accuracy = train_with_config(args)
            logger.info(f"Final model accuracy: {final_accuracy:.4f}")
            
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}", exc_info=True)
        raise


def run_standard_training(args) -> float:
    """
    Execute standard training with fixed hyperparameters.

    This function runs a single training session with the provided
    configuration, without hyperparameter search.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing all training parameters.

    Returns
    -------
    float
        Final validation accuracy achieved during training.

    See Also
    --------
    train_with_config : Core training workflow implementation
    """
    try:
        logger.info("Starting standard training mode")
        final_accuracy = train_with_config(args)
        logger.info(f"Training completed successfully with accuracy: {final_accuracy:.4f}")
        return final_accuracy
        
    except Exception as e:
        logger.error(f"Error during standard training: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    """
    Main entry point for CoLog training script.

    Parses command-line arguments and dispatches to either hyperparameter
    tuning or standard training mode based on configuration.
    
    The script supports two execution modes:
    1. Tuning mode: When --tuning flag is set, performs hyperparameter search
    2. Training mode: Standard training with fixed hyperparameters
    
    Random seeds are set before any training for reproducibility.
    """
    # Configure logging to write to colog_execution.log
    log_path = Path(constants.DIR_LOGS) / constants.FILE_COLOG_LOG
    log_dir = log_path.parent
    if log_dir and not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Parse command-line arguments
    parsed_args = parse_train_arguments()
    
    # Determine execution mode based on parsed arguments
    if isinstance(parsed_args, tuple):
        # Tuning mode: parsed_args is (config, args) tuple
        config, args = parsed_args
        
        logger.info(f"Tuning mode activated for dataset: {args.dataset}")
        logger.info(f"Random seed: {args.random_seed}")
        
        # Set random seeds for reproducibility
        set_random_seeds(args.random_seed)
        
        # Execute hyperparameter tuning
        run_hyperparameter_tuning(config, args)
        
    else:
        # Standard training mode: parsed_args is just args
        args = parsed_args
        
        logger.info(f"Standard training mode for dataset: {args.dataset}")
        logger.info(f"Random seed: {args.random_seed}")
        
        # Set random seeds for reproducibility
        set_random_seeds(args.random_seed)
        
        # Execute standard training
        final_accuracy = run_standard_training(args)

