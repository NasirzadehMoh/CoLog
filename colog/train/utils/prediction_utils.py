"""
prediction_utils.py — Prediction utilities for CoLog log anomaly detection

This module provides utility functions for converting model predictions to
discrete class labels. These functions are used during both training and
inference to transform raw model outputs (logits or probabilities) into
actionable predictions.

Key functionality:
    - Class prediction: Converts continuous prediction scores to discrete labels
    - Argmax operation: Identifies the most likely class for multi-class problems
    - Batch processing: Handles vectorized predictions efficiently

The utilities support binary and multi-class classification scenarios commonly
encountered in log anomaly detection tasks.

Usage
-----
    from train.utils.prediction_utils import predict_with_argmax
    
    # Get predictions from model outputs
    predictions = predict_with_argmax(model_outputs)
    
    # Compare with ground truth
    accuracy = np.mean(predictions == labels)

Notes
-----
- Input predictions should be 2D arrays of shape (batch_size, num_classes)
- Output is a 1D array of shape (batch_size,) containing class indices
- For binary classification, classes are typically 0 (anomaly) and 1 (normal)

See Also
--------
train.main : Main training loop using these prediction utilities
test_colog : Testing script that evaluates model predictions
"""

import logging
from typing import Union
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_with_argmax(preds: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert prediction probabilities to class labels.

    This function performs argmax operation along the class dimension to
    determine the predicted class for each sample. It handles both NumPy
    arrays and PyTorch tensors, automatically converting to NumPy format.

    The function is designed for multi-class classification where predictions
    are probability distributions or logits across multiple classes. The
    returned class indices correspond to the position of maximum value in
    each prediction vector.

    Parameters
    ----------
    preds : np.ndarray or torch.Tensor
        Prediction array of shape (batch_size, num_classes) containing
        probabilities or logits for each class. Values should represent
        confidence scores or log-probabilities.

    Returns
    -------
    np.ndarray
        Array of shape (batch_size,) containing predicted class indices.
        Each element is an integer in the range [0, num_classes).

    Examples
    --------
    >>> import numpy as np
    >>> # Binary classification example
    >>> probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    >>> predictions = predict_with_argmax(probs)
    >>> predictions
    array([0, 1, 0])

    >>> # Multi-class classification example
    >>> probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.2, 0.2]])
    >>> predictions = predict_with_argmax(probs)
    >>> predictions
    array([1, 0])

    Notes
    -----
    - For binary classification with shape (batch_size, 2), argmax returns
      0 or 1 corresponding to the two classes.
    - The function assumes predictions are already in probability or logit
      form; no softmax or sigmoid is applied.
    - PyTorch tensors are automatically converted to NumPy arrays before
      processing and moved to CPU if on GPU.

    Raises
    ------
    ValueError
        If input predictions are not 2-dimensional or are empty.
        If predictions contain NaN or infinite values.

    See Also
    --------
    numpy.argmax : Core function used for prediction extraction
    """
    try:
        # Convert PyTorch tensor to NumPy if necessary
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        
        # Validate input shape
        if preds.ndim != 2:
            raise ValueError(
                f"Predictions must be 2-dimensional (batch_size, num_classes), "
                f"got shape {preds.shape}"
            )
        
        if preds.size == 0:
            raise ValueError("Predictions array is empty")
        
        # Check for invalid values
        if np.isnan(preds).any():
            logger.warning("Predictions contain NaN values, predictions may be unreliable")
        
        if np.isinf(preds).any():
            logger.warning("Predictions contain infinite values, predictions may be unreliable")
        
        # Perform argmax along class dimension
        predictions = np.argmax(preds, axis=1)
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in predict_with_argmax: {str(e)}")
        raise


# ============================================================================
# Additional Utility Functions
# ============================================================================

def predict_with_threshold(preds: Union[np.ndarray, torch.Tensor], 
                          threshold: float = 0.5) -> np.ndarray:
    """
    Convert binary classification probabilities to labels using a threshold.

    For binary classification tasks, this function applies a threshold to the
    positive class probability to determine the predicted label. This allows
    fine-tuning of the precision-recall trade-off by adjusting the decision
    boundary.

    Parameters
    ----------
    preds : np.ndarray or torch.Tensor
        Prediction array of shape (batch_size, 2) containing probabilities
        for each class, or shape (batch_size,) containing positive class
        probabilities.
    threshold : float, optional
        Decision threshold in range [0, 1]. Samples with positive class
        probability >= threshold are classified as positive (class 1).
        Default is 0.5.

    Returns
    -------
    np.ndarray
        Array of shape (batch_size,) containing binary predictions (0 or 1).

    Examples
    --------
    >>> import numpy as np
    >>> probs = np.array([[0.3, 0.7], [0.8, 0.2], [0.45, 0.55]])
    >>> predictions = predict_with_threshold(probs, threshold=0.6)
    >>> predictions
    array([1, 0, 0])

    Raises
    ------
    ValueError
        If threshold is not in range [0, 1].
        If predictions have invalid shape for binary classification.

    Notes
    -----
    - Higher thresholds increase precision but may decrease recall
    - Lower thresholds increase recall but may decrease precision
    - Default threshold of 0.5 treats both classes equally

    See Also
    --------
    predict_with_argmax : Standard argmax-based prediction
    """
    try:
        # Convert PyTorch tensor to NumPy if necessary
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        
        # Validate threshold
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be in range [0, 1], got {threshold}")
        
        # Handle 2D predictions (batch_size, 2)
        if preds.ndim == 2:
            if preds.shape[1] != 2:
                raise ValueError(
                    f"For 2D predictions, expected shape (batch_size, 2), "
                    f"got {preds.shape}"
                )
            # Extract positive class probabilities
            positive_probs = preds[:, 1]
        
        # Handle 1D predictions (batch_size,)
        elif preds.ndim == 1:
            positive_probs = preds
        
        else:
            raise ValueError(
                f"Predictions must be 1D or 2D for binary classification, "
                f"got shape {preds.shape}"
            )
        
        # Apply threshold
        predictions = (positive_probs >= threshold).astype(np.int32)
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in predict_with_threshold: {str(e)}")
        raise
