"""
test.py — Evaluation and testing pipeline for CoLog collaborative transformer

This module implements the complete evaluation pipeline for trained CoLog models,
supporting both single-model evaluation and checkpoint ensembling for improved
prediction accuracy. It provides comprehensive metrics visualization and cross-dataset
generalizability testing capabilities.

Key features:
    - Single-model evaluation: Assess performance of individual checkpoints
    - Checkpoint ensembling: Combine predictions from multiple top-performing models
    - Comprehensive metrics: Generate confusion matrices, ROC curves, PR curves
    - Generalizability testing: Evaluate model performance on unseen datasets
    - Full report generation: Export detailed classification metrics to CSV
    - Visualization: High-quality SVG and EPS figures for publication

Evaluation modes:
    1. Standard evaluation: Test a single best checkpoint on test/validation sets
    2. Ensemble evaluation: Average predictions from top-k checkpoints
    3. Generalizability testing: Cross-dataset evaluation to assess transfer learning

Usage examples:
    Standard evaluation:
        python test.py --checkpoints-path runs/hadoop/model/
    
    Ensemble evaluation with top 5 checkpoints:
        python test.py --checkpoints-path runs/hadoop/model/ --ensemble True --num-ckpts 5
    
    Generalizability testing (train on hadoop, test on spark):
        python test.py --checkpoints-path runs/hadoop/model/ --eval-generalizability True \
            --name spark --dataset spark context_1
    
    Evaluate both validation and test sets:
        python test.py --checkpoints-path runs/hadoop/model/ --eval-sets valid_set test_set

Files produced:
    - confusion_matrix.svg/.eps: Visual confusion matrix
    - confusion_matrix_normalized.svg/.eps: Normalized confusion matrix
    - roc_curve.svg/.eps: Receiver Operating Characteristic curve
    - precision_recall_curve.svg/.eps: Precision-Recall curve
    - df_report.csv: Standard classification report (precision, recall, F1)
    - df_report_imb.csv: Imbalanced classification metrics

Dependencies:
    torch, numpy, pandas, matplotlib, scikit-learn, train.Trainer,
    neuralnetwork.CollaborativeTransformer, utils.BinaryClassificationMetrics
"""

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party imports
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

# Local imports
from neuralnetwork import CollaborativeTransformer, TextEmbedding
from train import Trainer
from train.utils import GroundTruthLoader, predict_with_argmax
from utils import parse_test_arguments
from utils import constants
from utils.metrics import ClassificationMetrics
from utils import constants

# Ignore all warnings (comment out during debugging if needed)
warnings.filterwarnings("ignore", category=UserWarning)

# Module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Prevent 'No handler could be found' warnings if imported before logging configured
    logger.addHandler(logging.NullHandler())


def load_checkpoints(checkpoints_path: str, num_ckpts: int) -> List[str]:
    """
    Load and sort checkpoint paths from the specified directory.

    Parameters
    ----------
    checkpoints_path : str
        Directory path containing model checkpoints.
    num_ckpts : int
        Maximum number of checkpoints to load.

    Returns
    -------
    List[str]
        Sorted list of checkpoint file paths (newest first).
    """
    checkpoint_pattern = str(Path(checkpoints_path) / 'best*')
    checkpoints = sorted(glob.glob(checkpoint_pattern), reverse=True)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_path}")
    
    return checkpoints[:num_ckpts]


def load_model_configuration(checkpoint_path: str) -> torch.nn.Module:
    """
    Load model arguments from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file.

    Returns
    -------
    argparse.Namespace
        Configuration arguments stored in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'args' not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} missing 'args' key")
    
    return checkpoint['args']


def verify_checkpoint_configurations(checkpoints: List[str], ensemble: bool) -> List[str]:
    """
    Verify that all checkpoint configurations match when using ensemble mode.
    If mismatches are found, group checkpoints by configuration and let user choose.

    Parameters
    ----------
    checkpoints : List[str]
        List of checkpoint file paths to verify.
    ensemble : bool
        Whether ensemble mode is enabled.

    Returns
    -------
    List[str]
        List of checkpoints to use (either all if matched, or user's selection).

    Raises
    ------
    ValueError
        If ensemble is enabled and checkpoint configurations don't match.
    """
    if not ensemble or len(checkpoints) <= 1:
        return checkpoints
    
    print("\n" + "-"*96)
    print(f"  🔍 Verifying configuration consistency across {len(checkpoints)} checkpoints...")
    print("-"*96)
    
    # Critical configuration attributes that must match for ensemble
    critical_attrs = [
        'd_model', 'n_heads', 'n_layers', 'dropout',
        'd_ff', 'max_seq_len', 'dataset', 'name'
    ]
    
    # Group checkpoints by configuration
    config_groups = {}
    
    for checkpoint_path in checkpoints:
        current_args = load_model_configuration(checkpoint_path)
        
        # Create configuration signature
        config_signature = tuple(
            getattr(current_args, attr, None) for attr in critical_attrs
        )
        
        if config_signature not in config_groups:
            config_groups[config_signature] = {
                'checkpoints': [],
                'config': {}
            }
            # Store readable configuration
            for attr in critical_attrs:
                if hasattr(current_args, attr):
                    config_groups[config_signature]['config'][attr] = getattr(current_args, attr)
        
        config_groups[config_signature]['checkpoints'].append(checkpoint_path)
    
    # Check if all checkpoints have the same configuration
    if len(config_groups) == 1:
        print(f"\n  ✓ All {len(checkpoints)} checkpoints have matching configurations.")
        print("="*96 + "\n")
        return checkpoints
    
    # Multiple configurations found - present options to user
    print("\n  ⚠️  Configuration mismatch detected in ensemble checkpoints!")
    print(f"  Found {len(config_groups)} different configurations:\n")
    
    # Display each configuration group
    group_list = list(config_groups.items())
    for i, (config_sig, group_data) in enumerate(group_list, start=1):
        print(f"Group {i}: ({len(group_data['checkpoints'])} checkpoints)")
        print(f"  Configuration:")
        for attr, value in group_data['config'].items():
            print(f"    {attr}: {value}")
        print(f"  Checkpoints:")
        for ckpt in group_data['checkpoints']:
            print(f"    - {Path(ckpt).name}")
        print()
    
    # Ask user to choose
    while True:
        try:
            choice = input(f"Select configuration group to use (1-{len(group_list)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                exit(0)
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(group_list):
                selected_group = group_list[choice_idx][1]
                selected_checkpoints = selected_group['checkpoints']
                print(f"\n✓ Selected Group {choice_idx + 1} with {len(selected_checkpoints)} checkpoints.\n")
                return selected_checkpoints
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(group_list)}.\n")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)


def initialize_data_loaders(
    args: object,
    eval_sets: List[str],
    train_dataset: GroundTruthLoader
) -> Dict[str, DataLoader]:
    """
    Initialize PyTorch DataLoaders for evaluation sets.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments containing batch size and other settings.
    eval_sets : List[str]
        List of evaluation set names (e.g., ['test_set', 'valid_set']).
    train_dataset : GroundTruthLoader
        Training dataset used to extract vocabulary and token mappings.

    Returns
    -------
    Dict[str, DataLoader]
        Dictionary mapping evaluation set names to their DataLoaders.
    """
    loaders = {}
    for eval_set in eval_sets:
        dataset = GroundTruthLoader(eval_set, args, train_dataset.tokens_dict)
        loaders[eval_set] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=constants.CPU_COUNT,
            pin_memory=True
        )
    
    return loaders


def initialize_models(
    args: object,
    vocab_size: int,
    embeddings: np.ndarray,
    num_classes: int
) -> Tuple[CollaborativeTransformer, TextEmbedding]:
    """
    Initialize CoLog collaborative transformer and embedding layer.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments for model architecture.
    vocab_size : int
        Size of the vocabulary.
    embeddings : np.ndarray
        Pre-trained word embeddings matrix.
    num_classes : int
        Number of output classes for classification.

    Returns
    -------
    Tuple[CollaborativeTransformer, TextEmbedding]
        Initialized model and embedding layer on the specified device.
    """
    # Initialize embedding layer
    embedding_layer = TextEmbedding(args, vocab_size, embeddings)
    
    # Initialize collaborative transformer
    cotransformer = CollaborativeTransformer(args, vocab_size, embeddings, num_classes=num_classes)
    
    # Move models to device
    cotransformer.to(args.device)
    embedding_layer.to(args.device)
    
    return cotransformer, embedding_layer


def compute_ensemble_predictions(
    ensemble_preds: Dict[str, List],
    loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble predictions by averaging individual model predictions.

    Parameters
    ----------
    ensemble_preds : Dict[str, List]
        Dictionary mapping sample IDs to lists of prediction probabilities.
    loader : DataLoader
        DataLoader containing ground truth labels and sample IDs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (averaged predictions, ground truth labels).
    """
    # Extract sample IDs and ground truth labels
    ids = [
        sample_id
        for batch_ids, _, _, _ in loader
        for sample_id in batch_ids
    ]
    answers = [
        np.array(answer)
        for _, _, _, batch_answers in loader
        for answer in batch_answers
    ]
    
    # Average predictions across all models
    average_predictions = np.array([
        np.mean(np.array(ensemble_preds[sample_id]), axis=0)
        for sample_id in ids
    ])
    
    return average_predictions, np.array(answers)


def save_classification_metrics(
    args: object,
    average_predictions: np.ndarray,
    answers: np.ndarray,
    labels: List[str],
    color_map: Dict,
    checkpoint_name: str = None,
    ensemble: bool = False
) -> None:
    """
    Generate and save comprehensive classification metrics and visualizations.

    This function creates confusion matrices (standard and normalized), ROC curves,
    precision-recall curves, and detailed classification reports. All figures are
    saved in both SVG and EPS formats for publication quality.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration arguments containing result path.
    average_predictions : np.ndarray
        Array of prediction probabilities of shape (n_samples, n_classes).
    answers : np.ndarray
        Array of ground truth labels.
    labels : List[str]
        List of class label names.
    color_map : Dict
        Color mapping for confusion matrix visualization.
    checkpoint_name : str, optional
        Name of the checkpoint being evaluated (default: None).
    ensemble : bool, optional
        Whether this is an ensemble evaluation (default: False).

    Returns
    -------
    None
    """
    print('\n  📈 Generating classification reports and visualizations...')
    start_time = time.time()
    
    # Initialize classification metrics
    classification_metrics = ClassificationMetrics(
        y_true=answers,
        y_preds_prob=average_predictions[:, 1],
        y_preds=predict_with_argmax(average_predictions),
        labels=labels
    )
    
    # Create result directory with subdirectories based on checkpoint and ensemble status
    result_path = Path(args.result_path)
    
    # Add checkpoint subdirectory
    if checkpoint_name:
        # Remove extension from checkpoint name
        checkpoint_dir = Path(checkpoint_name).stem
        result_path = result_path / checkpoint_dir
    
    # Add ensemble subdirectory if applicable
    if ensemble:
        result_path = result_path / 'ensembled'
    
    result_path.mkdir(parents=True, exist_ok=True)
    
    # Generate and save confusion matrix
    _save_confusion_matrix(
        classification_metrics,
        result_path,
        color_map,
        normalized=False
    )
    
    # Generate and save normalized confusion matrix
    _save_confusion_matrix(
        classification_metrics,
        result_path,
        color_map,
        normalized=True
    )
    
    # Generate and save ROC curve
    _save_roc_curve(classification_metrics, result_path)
    
    # Generate and save precision-recall curve
    _save_precision_recall_curve(classification_metrics, result_path)
    
    # Generate and save classification reports
    _save_classification_reports(classification_metrics, result_path)
    
    elapsed_time = time.time() - start_time
    print(f'  ✓ Report generation completed in {int(elapsed_time)}s')
    print(f'  💾 Results saved to: {result_path}')


def _save_confusion_matrix(
    metrics: ClassificationMetrics,
    result_path: Path,
    color_map: Dict,
    normalized: bool = False
) -> None:
    """
    Generate and save confusion matrix figure.

    Parameters
    ----------
    metrics : BinaryClassificationMetrics
        Metrics object containing confusion matrix data.
    result_path : Path
        Directory path for saving figures.
    color_map : Dict
        Color mapping for visualization.
    normalized : bool, optional
        Whether to normalize the confusion matrix (default: False).

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10, 8.4), num=1, clear=True, dpi=600, facecolor='#f0f0f0')
    
    # Plot confusion matrix
    metrics.plot_confusion_matrix(normalize=normalized, cmap=color_map)
    
    # Add separating lines
    plt.plot([-0.5, 1.5], [0.5, 0.5], color="#e34a6f", linewidth=5)
    plt.plot([0.5, 0.5], [-0.5, 1.5], color="#e34a6f", linewidth=5)
    
    # Save figure
    suffix = '_normalized' if normalized else ''
    fig.savefig(result_path / f'confusion_matrix{suffix}.svg')
    fig.savefig(result_path / f'confusion_matrix{suffix}.eps')
    
    plt.close(fig)


def _save_roc_curve(metrics: ClassificationMetrics, result_path: Path) -> None:
    """
    Generate and save ROC curve figure.

    Parameters
    ----------
    metrics : BinaryClassificationMetrics
        Metrics object containing ROC curve data.
    result_path : Path
        Directory path for saving figures.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10, 8.4), num=1, clear=True, dpi=600)
    metrics.plot_roc_curve()
    
    # Save figure
    fig.savefig(result_path / 'roc_curve.svg')
    fig.savefig(result_path / 'roc_curve.eps')
    
    plt.close(fig)


def _save_precision_recall_curve(
    metrics: ClassificationMetrics,
    result_path: Path
) -> None:
    """
    Generate and save precision-recall curve figure.

    Parameters
    ----------
    metrics : BinaryClassificationMetrics
        Metrics object containing precision-recall curve data.
    result_path : Path
        Directory path for saving figures.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(10, 8.4), num=1, clear=True, dpi=600)
    metrics.plot_precision_recall_curve()
    
    # Save figure
    fig.savefig(result_path / 'precision_recall_curve.svg')
    fig.savefig(result_path / 'precision_recall_curve.eps')
    
    plt.close(fig)


def _save_classification_reports(
    metrics: ClassificationMetrics,
    result_path: Path
) -> None:
    """
    Generate and save classification reports as CSV files.

    Parameters
    ----------
    metrics : BinaryClassificationMetrics
        Metrics object containing classification report data.
    result_path : Path
        Directory path for saving reports.

    Returns
    -------
    None
    """
    report, report_imb = metrics.print_report()
    
    # Convert to DataFrames and save
    df_report = pd.DataFrame(report).transpose()
    df_report_imb = pd.DataFrame(report_imb).transpose()
    
    df_report.to_csv(result_path / 'df_report.csv')
    df_report_imb.to_csv(result_path / 'df_report_imb.csv')


if __name__ == '__main__':
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
    args = parse_test_arguments()

    # Extract experiment configuration
    ensemble = args.ensemble
    num_ckpts = args.num_ckpts
    eval_sets = args.eval_sets
    plot_metrics = args.plot_metrics
    eval_generalizability = args.eval_generalizability
    name = args.name
    dataset = args.dataset

    # Load checkpoint paths
    try:
        checkpoints = load_checkpoints(args.checkpoints_path, num_ckpts)
    except FileNotFoundError as e:
        logger.error(str(e))
        print(f"Error: {e}")
        exit(1)

    # Verify checkpoint configurations match when using ensemble
    # Returns selected checkpoints if user chooses from grouped configurations
    checkpoints = verify_checkpoint_configurations(checkpoints, ensemble)
    
    # Update num_ckpts if user selected fewer checkpoints
    num_ckpts = min(num_ckpts, len(checkpoints))

    # Load model configuration from first checkpoint
    # Keep checkpoint args separate so we can load the original vocabulary
    # and embeddings used during training (required for generalizability tests)
    ckpt_args = load_model_configuration(checkpoints[0])

    # Prepare evaluation args: if generalizability testing is requested,
    # use a copy of checkpoint args for loading the model, but update
    # evaluation-specific fields (dataset/name) for the target dataset.
    from copy import deepcopy
    eval_args = deepcopy(ckpt_args)
    if eval_generalizability:
        eval_args.name = name
        eval_args.dataset = dataset
        eval_sets = ['test_set']
    else:
        # Standard evaluation uses the checkpoint args as-is
        eval_args = ckpt_args

    # Load color map for visualizations
    color_map_path = Path('utils/color_map.p')
    try:
        with open(color_map_path, 'rb') as handle:
            color_map = pickle.load(handle)
    except FileNotFoundError:
        logger.warning(f"Color map not found at {color_map_path}, using default colors")
        color_map = None

    # Display evaluation configuration
    print("\n" + "="*96)
    print(f"{'EVALUATION STARTED':^96}")
    print("="*96 + "\n")
    
    evaluation_sets = list(eval_sets)
    if len(evaluation_sets) > 1:
        set_names = ', '.join(s.split('_')[0] for s in evaluation_sets[:-1])
        print(f'  📋 Evaluating {set_names}, and {evaluation_sets[-1].split("_")[0]} sets')
    else:
        print(f'  📋 Evaluating {evaluation_sets[0].split("_")[0]} set')

    # Start evaluation timer
    start_time_eval = time.time()

    # Initialize datasets and data loaders
    # Use checkpoint args to construct the training dataset (vocab/embeddings
    # must match the model that was trained). Use eval_args for evaluation
    # loaders so evaluation dataset selection/name can differ (generalizability).
    train_dataset = GroundTruthLoader('train_set', ckpt_args)
    loaders = initialize_data_loaders(eval_args, evaluation_sets, train_dataset)

    # Define class labels
    labels = ["0: Anomaly", "1: Normal"]

    # Initialize models using the checkpoint configuration (so embedding
    # dimensions and vocab_size match the saved weights)
    cotransformer, embedding_layer = initialize_models(
        ckpt_args,
        train_dataset.vocab_size,
        train_dataset.embeddings,
        train_dataset.num_classes
    )

    # Initialize trainer with evaluation args (device, batch_size, result paths)
    trainer = Trainer(eval_args)

    # Initialize ensemble tracking
    max_accuracy = 0
    ensemble_preds = {eval_set: {} for eval_set in evaluation_sets}
    ensemble_accuracies = {eval_set: [] for eval_set in evaluation_sets}

    # Display model information
    num_params = sum(p.numel() for p in cotransformer.parameters()) / 1e6
    print(f"  🤖 Model parameters: {num_params:.2f}M")
    
    # Display evaluation mode
    if ensemble and num_ckpts > 1:
        print(f"  🔗 Ensembling mode: {num_ckpts} checkpoints")
    else:
        print('  🎯 Single model evaluation')
    print("\n" + "-"*96)

    # Iterate over checkpoints for evaluation/ensembling
    for i, checkpoint in enumerate(checkpoints):
        if i >= num_ckpts:
            break

        checkpoint_name = Path(checkpoint).name
        if ensemble and num_ckpts > 1:
            print(f'\n  🔄 Processing checkpoint {i+1}/{num_ckpts}: {checkpoint_name}')
        else:
            print(f'\n  📦 Loading checkpoint: {checkpoint_name}')

        # Load model state
        state_dict = torch.load(checkpoint, weights_only=False)['state_dict']
        cotransformer.load_state_dict(state_dict)

        # Evaluate on each evaluation set
        for eval_set in evaluation_sets:
            print(f'\n  ⚡ Evaluating on {eval_set.split("_")[0]} set...')
            
            # Perform evaluation
            accuracy, preds = trainer.evaluate_model(
                cotransformer,
                embedding_layer,
                loaders[eval_set]
            )
            
            print()
            print(f'  📊 Accuracy: {accuracy:.2f}%')

            # Accumulate predictions for ensembling
            if ensemble and num_ckpts > 1:
                for sample_id, pred in preds.items():
                    if sample_id not in ensemble_preds[eval_set]:
                        ensemble_preds[eval_set][sample_id] = []
                    ensemble_preds[eval_set][sample_id].append(pred)
            else:
                for sample_id, pred in preds.items():
                    ensemble_preds[eval_set][sample_id] = [pred]

            # Compute ensemble accuracy
            average_predictions, answers = compute_ensemble_predictions(
                ensemble_preds[eval_set],
                loaders[eval_set]
            )

            if ensemble and num_ckpts > 1:
                # Compute accuracy for current ensemble
                accuracy = np.mean(
                    predict_with_argmax(average_predictions) == answers
                ) * 100
                improvement = accuracy - ensemble_accuracies[eval_set][-1] if ensemble_accuracies[eval_set] else 0
                arrow = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f'  {arrow} Ensemble accuracy ({i+1} models): {accuracy:.2f}%')

            ensemble_accuracies[eval_set].append(accuracy)

            # Generate full report for test set at appropriate time
            should_generate_report = (
                eval_set == 'test_set' and
                plot_metrics and
                (
                    (not ensemble and max_accuracy < accuracy) or
                    (ensemble and i + 1 == num_ckpts)
                )
            )

            if should_generate_report:
                # Print evaluation timing
                elapsed_time_eval = time.time() - start_time_eval
                print(f'\n  ⏱️  Test set evaluation completed in {int(elapsed_time_eval)}s')

                # Generate and save comprehensive metrics
                if color_map is not None:
                    save_classification_metrics(
                        args,
                        average_predictions,
                        answers,
                        labels,
                        color_map,
                        checkpoint_name=checkpoint_name if not ensemble or num_ckpts == 1 else None,
                        ensemble=(ensemble and num_ckpts > 1)
                    )
                else:
                    logger.warning("Skipping metric plots due to missing color map")

                max_accuracy = accuracy

    # Print final results summary
    total_time = time.time() - start_time_eval
    print('\n' + '='*96)
    print(f"{'EVALUATION SUMMARY':^96}")
    print('='*96)
    
    for eval_set in eval_sets:
        max_acc = max(ensemble_accuracies[eval_set])
        
        if ensemble and num_ckpts > 1:
            print(f"  🏆 Maximum ensemble accuracy for {eval_set.split('_')[0]} set: {max_acc:.2f}%")
        elif not ensemble and num_ckpts > 1:
            print(f"  🏆 Maximum evaluation accuracy for {eval_set.split('_')[0]} set: {max_acc:.2f}%")
        else:
            print(f"  🎯 Overall evaluation accuracy for {eval_set.split('_')[0]} set: {max_acc:.2f}%")
    
    print(f"\n  ⏱️  Total evaluation time: {int(total_time)}s ({total_time/60:.1f} min)")
    print('='*96 + '\n')
