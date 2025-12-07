"""
main.py — Training pipeline for CoLog collaborative transformer model

This module implements the complete training and evaluation pipeline for the
CoLog collaborative transformer architecture. It provides functionality for
standard training, hyperparameter tuning with Ray Tune, and comprehensive
evaluation of log anomaly detection models.

Key components:
    - Trainer class: Orchestrates model training, evaluation, and tuning
    - Training loop: Implements gradient-based optimization with early stopping
        and learning rate decay
    - Evaluation: Computes accuracy metrics on validation and test sets
    - Hyperparameter tuning: Integrates with Ray Tune for automated search
    - Checkpointing: Saves best models based on validation performance
    - Progress tracking: Real-time training progress with loss, accuracy, and
        timing information

Training features:
    - Early stopping: Halts training when validation accuracy stops improving
    - Learning rate decay: Reduces learning rate when accuracy plateaus
    - Gradient clipping: Prevents gradient explosion during training
    - Multi-GPU support: Automatic data parallelism with DataParallel
    - Comprehensive logging: Tracks all training metrics to file

Usage examples:
    Standard training:
        trainer = Trainer(args)
        trainer.train_model(model, embedding_layer, train_loader, eval_loader)
    
    Hyperparameter tuning:
        trainer = Trainer(args)
        trainer.tune_model(config, checkpoint_dir)
    
    Model evaluation:
        accuracy, predictions = trainer.evaluate_model(model, embeddings, loader)

Files produced:
    - training_metrics.log: Complete training history with loss, accuracy, timing
    - best<random_seed>.pkl: Best model checkpoint during training
    - best<accuracy>_<random_seed>.pkl: Final renamed checkpoint with accuracy

Dependencies:
    torch, numpy, ray (for tuning), train.utils.groundtruth_loader,
    neuralnetwork.collaborative_transformer_tunable, train.utils.prediction_utils
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Third-party imports
import numpy as np
import ray
from ray import tune
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from neuralnetwork import CollaborativeTransformerTunable
from train.utils import GroundTruthLoader
from train.utils import predict_with_argmax
from train.utils import constants

# Ignore all warnings (comment out during debugging if needed)
warnings.filterwarnings("ignore", category=UserWarning)

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent 'No handler could be found' warnings if imported before logging configured
    logger.addHandler(logging.NullHandler())

class Trainer:
    """
    Trainer orchestrates model training, evaluation, and hyperparameter tuning.

    This class encapsulates the complete training pipeline for the CoLog
    collaborative transformer model. It handles standard supervised learning,
    early stopping, learning rate scheduling, checkpoint management, and
    integration with Ray Tune for hyperparameter optimization.

    The training process includes:
        1. Iterative optimization over multiple epochs
        2. Gradient-based updates with configurable learning rate
        3. Periodic evaluation on validation set
        4. Early stopping when validation accuracy plateaus
        5. Learning rate decay for improved convergence
        6. Gradient norm clipping for training stability
        7. Best model checkpointing based on validation performance

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing training hyperparameters and settings:
        - max_epoch (int): Maximum number of training epochs
        - learning_rate (float): Initial learning rate for optimizer
        - batch_size (int): Batch size for training and evaluation
        - device (str): Device for computation ('cuda' or 'cpu')
        - result_path (str): Directory to save training logs
        - checkpoint_path (str): Directory to save model checkpoints
        - eval_start (int): Epoch to start validation evaluation
        - early_stop (int): Number of epochs without improvement before stopping
        - lr_decay (float): Learning rate decay factor
        - decay_times (int): Maximum number of learning rate decays
        - grad_norm_clip (float): Gradient norm clipping threshold (0 to disable)
        - loss_fn (callable): Loss function for optimization
        - seed (int): Random seed for reproducibility

    Attributes
    ----------
    args : argparse.Namespace
        Stored configuration object.

    Methods
    -------
    train_model(model, embedding_layer, train_loader, eval_loader)
        Execute complete training loop with early stopping and checkpointing.
    evaluate_model(model, embedding_layer, eval_loader)
        Evaluate model performance on validation/test data.
    tune_model(config, checkpoint_dir)
        Train model with Ray Tune hyperparameter configuration.
    evaluate_tunable_model(model_tunable, eval_loader)
        Evaluate tunable model variant during hyperparameter search.

    Notes
    -----
    - The trainer automatically saves the best model based on validation accuracy
    - Training progress is logged to both console and file
    - Early stopping prevents overfitting by monitoring validation performance
    - Learning rate decay helps fine-tune convergence in later epochs
    - Gradient clipping prevents gradient explosion in deep networks

    Examples
    --------
    >>> from train.main import Trainer
    >>> trainer = Trainer(args)
    >>> accuracies = trainer.train_model(model, embeddings, train_dl, val_dl)
    >>> print(f"Final validation accuracy: {accuracies[-1]:.2f}%")
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize the Trainer with configuration.
        
        Parameters
        ----------
        args : argparse.Namespace
            Training configuration and hyperparameters.
        """
        self.args = args
        logger.debug(f"Initialized Trainer with args: {args}")

    
    def train_model(
        self,
        model: nn.Module,
        embedding_layer: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader
    ) -> float:
        """
        Execute complete training loop with early stopping and checkpointing.

        This method implements the main training pipeline including forward pass,
        backward propagation, optimization, validation, learning rate scheduling,
        and early stopping. Training progress is logged to both console and file.

        The training loop performs:
            1. Forward pass through model with input messages and sequences
            2. Loss computation using cross-entropy
            3. Backward propagation and gradient computation
            4. Gradient clipping (if enabled)
            5. Parameter updates via optimizer
            6. Periodic validation and checkpoint saving
            7. Early stopping when validation plateaus

        Parameters
        ----------
        model : nn.Module
            The collaborative transformer model to train.
        embedding_layer : nn.Module
            Embedding layer that converts input tokens to dense vectors.
        train_loader : DataLoader
            DataLoader providing training batches (log_id, messages, sequences, labels).
        eval_loader : DataLoader
            DataLoader providing validation batches for periodic evaluation.

        Returns
        -------
        float
            Best validation accuracy achieved during training.

        Notes
        -----
        - Best model is saved when validation accuracy improves
        - Learning rate decays when accuracy plateaus (up to decay_times)
        - Training stops early if no improvement for early_stop epochs
        - All metrics are logged to training_metrics.log file
        - Progress printed to console includes loss, learning rate, and time estimates

        See Also
        --------
        evaluate_model : Evaluation logic for computing accuracy
        """
        try:
            logger.info("Starting model training")
            time_start_train = time.time()

            # Initialize training metrics log file
            metrics_log_path = Path(self.args.checkpoint_path) / constants.FILE_TRAINING_LOG
            
            with open(metrics_log_path, 'w+') as training_metrics_file:
                training_metrics_file.write(f'Arguments: {str(self.args)}\n\n\n\n')
                logger.debug(f"Training metrics will be saved to {metrics_log_path}")

                # Initialize training state
                loss_sum = 0
                best_eval_accuracy = 0.0
                early_stop_counter = 0
                lr_decay_counter = 0

                # Configure optimizer with parsed parameters
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=self.args.learning_rate,
                    **self.args.optimizer_params
                )
                logger.info(f"Optimizer: Adam with learning rate={self.args.learning_rate}, params={self.args.optimizer_params}")

                loss_fn = self.args.loss_fn
                final_accuracy = []

                # Main training loop
                print("\n" + "="*96)
                print(f"{'TRAINING STARTED':^96}")
                print("="*96 + "\n")
                
                for epoch in range(0, self.args.max_epoch):
                    logger.debug(f"Starting epoch {epoch + 1}/{self.args.max_epoch}")
                    time_start_epoch = time.time()
                    
                    # Print epoch header
                    print("\n" + "-"*96)
                    print(f" Epoch [{epoch + 1}/{self.args.max_epoch}]")
                    print("-"*96)

                    for step, (
                            log_id,
                            input_messages,
                            input_sequences,
                            target_labels,
                    ) in enumerate(train_loader):

                        batch_loss = 0
                        optimizer.zero_grad()

                        input_messages = input_messages.to(self.args.device)
                        input_sequences = input_sequences.to(self.args.device)
                        target_labels = target_labels.to(self.args.device)

                        embeddings = embedding_layer(input_messages)

                        predictions = model(input_messages, embeddings, input_sequences)
                        loss = loss_fn(predictions, target_labels)
                        loss.backward()

                        loss_sum += loss.cpu().data.numpy()
                        batch_loss += loss.cpu().data.numpy()

                        # Display training progress with real-time metrics
                        steps_total = len(train_loader)
                        current_lr = optimizer.param_groups[0]['lr']
                        time_per_step = (time.time() - time_start_epoch) / (step + 1)
                        remaining_steps = steps_total - (step + 1)
                        remaining_time_min = int((time_per_step * remaining_steps) / 60)
                        remaining_time_sec = int((time_per_step * remaining_steps) % 60)
                        
                        # Calculate progress percentage
                        progress_pct = min((step + 1) / steps_total * 100, 100.0)
                        bar_length = 40
                        filled_length = int(bar_length * (step + 1) // steps_total)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        
                        print(
                            f"\r  [{bar}] {progress_pct:5.1f}% | "
                            f"Loss: {batch_loss / self.args.batch_size:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"ETA: {remaining_time_min:02d}:{remaining_time_sec:02d}",
                            end=''
                        )

                        # Gradient norm clipping
                        if self.args.grad_norm_clip > 0:
                            nn.utils.clip_grad_norm_(
                                model.parameters(),
                                self.args.grad_norm_clip
                            )

                        optimizer.step()

                    time_end_epoch = time.time()
                    elapse_time_epoch = time_end_epoch - time_start_epoch
                    epoch_finish = epoch + 1
                    avg_epoch_loss = loss_sum / len(train_loader.dataset)
                    
                    # Print epoch summary
                    print(f"\n  ✓ Epoch {epoch_finish} completed in {int(elapse_time_epoch)}s | Avg Loss: {avg_epoch_loss:.6f}")
                    logger.info(f"Epoch {epoch_finish} completed in {int(elapse_time_epoch)}s, "
                            f"avg loss: {avg_epoch_loss:.4f}")

                    # Write epoch metrics to log file
                    avg_loss = loss_sum / len(train_loader.dataset)
                    current_lr = optimizer.param_groups[0]['lr']
                    speed_per_batch = elapse_time_epoch / (step + 1) if step > 0 else 0
                    
                    training_metrics_file.write(
                        f'Epoch: {epoch_finish}, Loss: {avg_loss:.6f}, '
                        f'Lr: {current_lr:.6e}\n'
                        f'Elapsed time: {int(elapse_time_epoch)} sec, '
                        f'Speed(s/batch): {speed_per_batch:.4f}\n'
                    )
                    
                    # Validation evaluation
                    if epoch_finish >= self.args.evaluation_start:
                        print('\n  ⚡ Starting validation...')
                        logger.debug("Starting validation evaluation")
                        
                        time_start_val = time.time()
                        accuracy, _ = self.evaluate_model(model, embedding_layer, eval_loader)
                        time_end_val = time.time()
                        elapsed_time_val = time_end_val - time_start_val
                        
                        print(f' Completed in {int(elapsed_time_val)}s')
                        print(f'  📊 Validation Accuracy: {accuracy:.2f}%')
                        logger.info(f"Validation accuracy: {accuracy:.2f}% (took {int(elapsed_time_val)}s)")
                        
                        training_metrics_file.write(f'Evaluation Accuracy: {accuracy:.4f}\n\n')
                        final_accuracy.append(accuracy)
                        
                        if accuracy > best_eval_accuracy:
                            # Save new best model
                            improvement = accuracy - best_eval_accuracy
                            print(f'  🎯 NEW BEST! Improved by {improvement:.2f}% (prev: {best_eval_accuracy:.2f}%)')
                            logger.info(f"New best accuracy: {accuracy:.2f}% (previous: {best_eval_accuracy:.2f}%)")
                            checkpoint_path = Path(self.args.checkpoint_path) / f'{constants.CHECKPOINT_PREFIX_BEST}{self.args.random_seed}{constants.EXT_PKL}'
                            state = {
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'args': self.args,
                            }
                            torch.save(state, checkpoint_path)
                            print(f'  💾 Model checkpoint saved')
                            logger.debug(f"Saved best model to {checkpoint_path}")
                            
                            best_eval_accuracy = accuracy
                            early_stop_counter = 0

                        elif lr_decay_counter < self.args.decay_times:
                            # Apply learning rate decay
                            print(f'  ⚠️  No improvement - Applying learning rate decay ({lr_decay_counter + 1}/{self.args.decay_times})')
                            logger.info(f"Learning rate decay triggered (count: {lr_decay_counter + 1}/{self.args.decay_times})")
                            
                            lr_decay_counter += 1
                            
                            # Reload best model
                            best_model_path = Path(self.args.checkpoint_path) / f'{constants.CHECKPOINT_PREFIX_BEST}{self.args.random_seed}{constants.EXT_PKL}'
                            model.load_state_dict(
                                torch.load(best_model_path, weights_only=False)['state_dict']
                            )
                            logger.debug(f"Reloaded best model from {best_model_path}")
                            
                            # Decay learning rate
                            old_lr = optimizer.param_groups[0]['lr']
                            for group in optimizer.param_groups:
                                group['lr'] *= self.args.lr_decay
                            new_lr = optimizer.param_groups[0]['lr']
                            print(f'  📉 Learning rate: {old_lr:.2e} → {new_lr:.2e}')
                            logger.info(f"Learning rate decayed: {old_lr:.6e} -> {new_lr:.6e}")

                        else:
                            # Check for early stopping
                            early_stop_counter += 1
                            print(f'  ⏸️  No improvement ({early_stop_counter}/{self.args.early_stop} epochs)')
                            logger.debug(f"No improvement: early stop counter = {early_stop_counter}/{self.args.early_stop}")
                            
                            if early_stop_counter == self.args.early_stop:
                                print("\n" + "="*96)
                                print(f"{'EARLY STOPPING TRIGGERED':^96}")
                                print("="*96)
                                print(f'  🏁 Best Validation Accuracy: {best_eval_accuracy:.2f}%')
                                logger.info(f"Early stopping triggered after {self.args.early_stop} epochs without improvement")
                                logger.info(f"Best validation accuracy achieved: {best_eval_accuracy:.2f}%")
                                
                                training_metrics_file.write(
                                    f'\n\nEarly stop reached. ---- '
                                    f'Best evaluation accuracy: {best_eval_accuracy:.4f}'
                                )
                                
                                # Rename checkpoint with accuracy in filename
                                old_checkpoint = Path(self.args.checkpoint_path) / f'{constants.CHECKPOINT_PREFIX_BEST}{self.args.random_seed}{constants.EXT_PKL}'
                                new_checkpoint = Path(self.args.checkpoint_path) / f'{constants.CHECKPOINT_PREFIX_BEST}{best_eval_accuracy:.2f}_{self.args.random_seed}{constants.EXT_PKL}'
                                old_checkpoint.rename(new_checkpoint)
                                logger.info(f"Renamed checkpoint to {new_checkpoint}")
                                
                                time_end_train = time.time()
                                elapse_time_train = time_end_train - time_start_train
                                hours = int(elapse_time_train // 3600)
                                minutes = int((elapse_time_train % 3600) // 60)
                                seconds = int(elapse_time_train % 60)
                                print(f'  ⏱️  Total Training Time: {hours}h {minutes}m {seconds}s')
                                print("="*96 + "\n")
                                logger.info(f"Total training time: {int(elapse_time_train)}s")
                                
                                training_metrics_file.write(
                                    f'\nTraining finished in: {int(elapse_time_train)} sec'
                                )
                                return best_eval_accuracy

                    loss_sum = 0
                
                # Training completed all epochs without early stopping
                time_end_train = time.time()
                elapse_time_train = time_end_train - time_start_train
                hours = int(elapse_time_train // 3600)
                minutes = int((elapse_time_train % 3600) // 60)
                seconds = int(elapse_time_train % 60)
                
                print("\n" + "="*96)
                print(f"{'TRAINING COMPLETED':^96}")
                print("="*96)
                print(f'  🏁 All {self.args.max_epoch} epochs completed')
                print(f'  🎯 Best Validation Accuracy: {best_eval_accuracy:.2f}%')
                print(f'  ⏱️  Total Training Time: {hours}h {minutes}m {seconds}s')
                print("="*96 + "\n")
                
                logger.info(f"Training completed all {self.args.max_epoch} epochs")
                logger.info(f"Best validation accuracy achieved: {best_eval_accuracy:.2f}%")
                logger.info(f"Total training time: {int(elapse_time_train)}s")
                
                # Rename checkpoint with accuracy in filename
                old_checkpoint = Path(self.args.checkpoint_path) / f'{constants.CHECKPOINT_PREFIX_BEST}{self.args.random_seed}{constants.EXT_PKL}'
                new_checkpoint = Path(self.args.checkpoint_path) / f'{constants.CHECKPOINT_PREFIX_BEST}{best_eval_accuracy:.2f}_{self.args.random_seed}{constants.EXT_PKL}'
                if old_checkpoint.exists():
                    old_checkpoint.rename(new_checkpoint)
                    logger.info(f"Renamed checkpoint to {new_checkpoint}")
                
                training_metrics_file.write(
                    f'\n\nTraining completed all epochs. Best evaluation accuracy: {best_eval_accuracy:.4f}\n'
                    f'Total training time: {hours}h {minutes}m {seconds}s\n'
                    f'Training finished in: {int(elapse_time_train)} sec'
                )
                return best_eval_accuracy
        
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

    def evaluate_model(
        self,
        model: nn.Module,
        embedding_layer: nn.Module,
        eval_loader: DataLoader
    ) -> Tuple[float, Dict[Any, np.ndarray]]:
        """
        Evaluate the collaborative transformer model on validation or test data.

        This method computes model accuracy by running inference on all batches
        in the evaluation data loader. It sets the model to evaluation mode,
        disables gradient computation, and collects predictions for analysis.

        Parameters
        ----------
        model : nn.Module
            The collaborative transformer model to evaluate.
        embedding_layer : nn.Module
            Embedding layer that converts input tokens to dense vectors.
        eval_loader : DataLoader
            DataLoader providing evaluation batches (log_id, messages, sequences, labels).

        Returns
        -------
        accuracy : float
            Overall accuracy as a percentage (0-100).
        predictions_dict : Dict[Any, np.ndarray]
            Dictionary mapping log IDs to their prediction logits.

        Notes
        -----
        - Model is temporarily set to eval mode, then restored to train mode
        - Predictions are argmax-decoded for accuracy computation
        - All predictions are saved for potential further analysis
        - Gradients are not computed during evaluation

        See Also
        --------
        train_model : Training loop that calls this method for validation
        """
        try:
            accuracy = []
            model.train(False)
            predictions_dict = {}
            
            logger.debug(f"Evaluating on {len(eval_loader.dataset)} samples")
            
            total_steps = len(eval_loader)
            
            for step, (
                    log_id,
                    input_messages,
                    input_sequences,
                    target_labels,
            ) in enumerate(eval_loader):
                input_messages = input_messages.to(self.args.device)
                input_sequences = input_sequences.to(self.args.device)

                embeddings = embedding_layer(input_messages)

                batch_predictions = model(input_messages, embeddings, input_sequences).cpu().data.numpy()

                target_labels = target_labels.cpu().data.numpy()
                accuracy += list(predict_with_argmax(batch_predictions) == target_labels)
                    
                # Save predictions
                for current_id, prediction in zip(log_id, batch_predictions):
                    predictions_dict[current_id] = prediction
                
                # Print validation progress
                progress_pct = min((step + 1) / total_steps * 100, 100.0)
                bar_length = 40
                filled_length = int(bar_length * (step + 1) // total_steps)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"\r  [{bar}] {progress_pct:5.1f}%", end='')

            model.train(True)
            final_accuracy = 100 * np.mean(np.array(accuracy))
            logger.debug(f"Evaluation completed: accuracy = {final_accuracy:.2f}%")
            
            return final_accuracy, predictions_dict
        
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            model.train(True)  # Ensure model is back in train mode
            raise

    def evaluate_tunable_model(
        self,
        model_tunable: nn.Module,
        eval_loader: DataLoader
    ) -> Tuple[float, Dict[Any, np.ndarray]]:
        """
        Evaluate the tunable collaborative transformer model.

        This method evaluates the tunable variant of the CoTransformer model
        during hyperparameter optimization with Ray Tune. The tunable model
        has embeddings built-in, so it doesn't require a separate embedding layer.

        Parameters
        ----------
        model_tunable : nn.Module
            The tunable collaborative transformer model with embedded embeddings.
        eval_loader : DataLoader
            DataLoader providing evaluation batches (log_id, messages, sequences, labels).

        Returns
        -------
        accuracy : float
            Overall accuracy as a percentage (0-100).
        predictions_dict : Dict[Any, np.ndarray]
            Dictionary mapping log IDs to their prediction logits.

        Notes
        -----
        - Similar to evaluate_model but for tunable model variant
        - Model is temporarily set to eval mode, then restored to train mode
        - No embedding layer parameter needed as embeddings are internal
        - Used exclusively during Ray Tune hyperparameter optimization

        See Also
        --------
        evaluate_model : Evaluation for standard model
        tune_model : Hyperparameter tuning that calls this method
        """
        try:
            accuracy = []
            model_tunable.train(False)
            predictions_dict = {}
            
            logger.debug(f"Evaluating tunable model on {len(eval_loader.dataset)} samples")
            
            for step, (
                    log_id,
                    input_messages,
                    input_sequences,
                    target_labels,
            ) in enumerate(eval_loader):
                input_messages = input_messages.to(self.args.device)
                input_sequences = input_sequences.to(self.args.device)
                
                batch_predictions = model_tunable(
                    input_messages, input_sequences
                ).cpu().data.numpy()

                target_labels = target_labels.cpu().data.numpy()
                accuracy += list(predict_with_argmax(batch_predictions) == target_labels)
                    
                # Save predictions
                for current_id, prediction in zip(log_id, batch_predictions):
                    predictions_dict[current_id] = prediction

            model_tunable.train(True)
            final_accuracy = 100 * np.mean(np.array(accuracy))
            logger.debug(f"Tunable model evaluation: accuracy = {final_accuracy:.2f}%")
            
            return final_accuracy, predictions_dict
        
        except Exception as e:
            logger.error(f"Error during tunable model evaluation: {e}", exc_info=True)
            model_tunable.train(True)  # Ensure model is back in train mode
            raise


def tune_model(config: Dict[str, Any], checkpoint_dir: Optional[str] = None, args: Any = None) -> None:
    """
    Train model with Ray Tune hyperparameter configuration.

    This standalone function is designed to be called by Ray Tune during hyperparameter
    optimization. It trains the tunable CoTransformer variant with the
    provided hyperparameter configuration and reports validation accuracy
    back to Ray Tune for optimization.

    Parameters
    ----------
    config : Dict[str, Any]
        Hyperparameter configuration from Ray Tune containing:
        - layers (int): Number of transformer layers
        - heads (int): Number of attention heads
        - hidden_size (int): Hidden dimension size
        - dropout_rate (float): Dropout probability
        - batch_size (int): Batch size for training
    checkpoint_dir : str, optional
        Directory for loading checkpoints (Ray Tune feature). Default is None.
    args : Any
        Configuration object containing training parameters.

    Returns
    -------
    None
        Results are reported to Ray Tune via tune.report().

    Notes
    -----
    - This function changes the working directory to PROJECT_ROOT
    - Uses CoTransformer_Tune which has embedded embeddings
    - Automatically terminates if 100% accuracy is achieved
    - Reports accuracy to Ray Tune after each evaluation epoch
    - Does not save checkpoints (Ray Tune handles that)
    - Must be a module-level function for Ray Tune to pickle it

    See Also
    --------
    Trainer.train_model : Standard training without hyperparameter tuning
    """
    
    def evaluate_tunable_model(
        model_tunable: nn.Module,
        eval_loader: DataLoader,
        device: str
    ) -> Tuple[float, Dict[Any, np.ndarray]]:
        """
        Local evaluation function for tunable model.
        
        Parameters
        ----------
        model_tunable : nn.Module
            The tunable collaborative transformer model.
        eval_loader : DataLoader
            DataLoader for evaluation data.
        device : str
            Device to run evaluation on.
            
        Returns
        -------
        accuracy : float
            Overall accuracy as percentage (0-100).
        predictions_dict : Dict[Any, np.ndarray]
            Dictionary mapping log IDs to prediction logits.
        """
        try:
            accuracy = []
            model_tunable.train(False)
            predictions_dict = {}
            
            logger.debug(f"Evaluating tunable model on {len(eval_loader.dataset)} samples")
            
            for step, (
                    log_id,
                    input_messages,
                    input_sequences,
                    target_labels,
            ) in enumerate(eval_loader):
                input_messages = input_messages.to(device)
                input_sequences = input_sequences.to(device)
                
                batch_predictions = model_tunable(
                    input_messages, input_sequences
                ).cpu().data.numpy()

                target_labels = target_labels.cpu().data.numpy()
                accuracy += list(predict_with_argmax(batch_predictions) == target_labels)
                    
                # Save predictions
                for current_id, prediction in zip(log_id, batch_predictions):
                    predictions_dict[current_id] = prediction

            model_tunable.train(True)
            final_accuracy = 100 * np.mean(np.array(accuracy))
            logger.debug(f"Tunable model evaluation: accuracy = {final_accuracy:.2f}%")
            
            return final_accuracy, predictions_dict
        
        except Exception as e:
            logger.error(f"Error during tunable model evaluation: {e}", exc_info=True)
            model_tunable.train(True)  # Ensure model is back in train mode
            raise
    
    try:
        logger.info(f"Starting Ray Tune training with config: {config}")
        import os
        os.chdir(constants.PROJECT_ROOT)  # Change to project root directory
        
        # Initialize data loaders
        train_dataset = GroundTruthLoader(constants.SPLIT_TRAIN, args, print_hints=False)
        eval_dataset = GroundTruthLoader(
            constants.SPLIT_VALID, args, train_dataset.tokens_dict, print_hints=False
        )
        train_loader = DataLoader(
            train_dataset,
            eval(args.batch_size),
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )
        eval_loader = DataLoader(
            eval_dataset,
            eval(args.batch_size),
            num_workers=os.cpu_count(),
            pin_memory=True
        )
        logger.debug(f"Loaded {len(train_dataset)} train samples, {len(eval_dataset)} val samples")

        # Create tunable CoTransformer model
        # Use num_classes from dataset instead of args for flexibility
        model_tunable = CollaborativeTransformerTunable(
            config, args, train_dataset.vocab_size, train_dataset.embeddings,
            num_classes=train_dataset.num_classes
        )

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model_tunable = nn.DataParallel(model_tunable)
        model_tunable.to(args.device)

        loss_sum = 0

        # Configure optimizer with parsed parameters
        optimizer = torch.optim.Adam(
            model_tunable.parameters(), 
            lr=args.learning_rate,
            **args.optimizer_params
        )

        loss_fn = args.loss_fn
        
        # Training loop for hyperparameter tuning
        for epoch in range(0, args.max_epoch):
            logger.debug(f"Tune epoch {epoch + 1}/{args.max_epoch}")

            for step, (
                    log_id,
                    input_messages,
                    input_sequences,
                    target_labels,
            ) in enumerate(train_loader):

                batch_loss = 0
                optimizer.zero_grad()

                input_messages = input_messages.to(args.device)
                input_sequences = input_sequences.to(args.device)
                target_labels = target_labels.to(args.device)

                predictions = model_tunable(input_messages, input_sequences)
                loss = loss_fn(predictions, target_labels)
                loss.backward()

                loss_sum += loss.cpu().data.numpy()
                batch_loss += loss.cpu().data.numpy()

                # Gradient norm clipping
                if args.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        model_tunable.parameters(),
                        args.grad_norm_clip
                    )

                optimizer.step()

            epoch_finish = epoch + 1

            # Periodic validation for Ray Tune
            if epoch_finish >= args.evaluation_start:
                accuracy, _ = evaluate_tunable_model(model_tunable, eval_loader, args.device)
                logger.info(f"Tune epoch {epoch_finish}: accuracy = {accuracy:.2f}%")
                
                # Early termination if perfect accuracy achieved
                if accuracy == constants.RAY_PERFECT_ACCURACY:
                    tune.report({constants.RAY_METRIC_ACCURACY: accuracy})
                    logger.info("Perfect accuracy achieved, terminating tuning")
                    return
                
                tune.report({constants.RAY_METRIC_ACCURACY: accuracy})
                
            loss_sum = 0
    
    except Exception as e:
        logger.error(f"Error during Ray Tune training: {e}", exc_info=True)
        raise
