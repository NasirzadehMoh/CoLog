"""
metrics.py — Classification metrics and visualization utilities

This module provides comprehensive evaluation metrics and visualization tools
for binary and multiclass classification tasks in the CoLog log anomaly detection system.
It includes functions for plotting confusion matrices, ROC curves, and
precision-recall curves, along with detailed classification reports.

The ClassificationMetrics class serves as a unified interface for computing
and visualizing various performance metrics including:
    - Confusion matrices with optional normalization (binary and multiclass)
    - ROC curves with AUC scores and threshold visualization (binary, one-vs-rest for multiclass)
    - Precision-Recall curves with F-score iso-curves (binary, one-vs-rest for multiclass)
    - Classification reports with standard and imbalanced metrics (binary and multiclass)

Usage
-----
    from utils.metrics import ClassificationMetrics
    
    # Initialize with predictions and labels
    metrics = ClassificationMetrics(
        y_true=true_labels,
        y_preds_prob=prediction_probabilities,
        y_preds=predicted_labels,
        labels=['Normal', 'Anomaly']  # Or ['Class 0', 'Class 1', 'Class 2'] for multiclass
    )
    
    # Generate visualizations
    metrics.plot_confusion_matrix()
    fpr, tpr, thresh, auc_score = metrics.plot_roc_curve()  # One-vs-rest for multiclass
    prec, recall, thresh = metrics.plot_precision_recall_curve()  # One-vs-rest for multiclass
    
    # Print detailed reports
    report, report_imb = metrics.print_report()

Notes
-----
- All plotting methods use matplotlib and seaborn for visualization
- Threshold-based metrics can be customized via the threshold parameter
- F-score computations support configurable beta values for different
  precision-recall trade-offs
- For multiclass problems, ROC and PR curves use one-vs-rest approach for the last class
- Confusion matrices automatically adapt to the number of classes detected

See Also
--------
sklearn.metrics : Standard machine learning metrics
imblearn.metrics : Metrics for imbalanced datasets
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis, arange, argmin, linspace
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_curve,
    average_precision_score,
    classification_report
)
from imblearn.metrics import classification_report_imbalanced
from itertools import product
import seaborn as sns

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class ClassificationMetrics(object):
    """
    Classification metrics and visualization utilities for binary and multiclass problems.

    This class provides a comprehensive suite of evaluation metrics and
    visualization tools for both binary and multiclass classification problems.
    It supports confusion matrices, ROC curves, precision-recall curves, and detailed
    classification reports. For multiclass problems, ROC and precision-recall curves
    use a one-vs-rest approach for the last class label.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0, 1 for binary; 0, 1, 2, ... for multiclass).
    y_preds_prob : array-like
        Predicted probabilities for the positive class (binary) or last class (multiclass),
        in range [0, 1].
    y_preds : array-like
        Predicted labels matching the format of y_true.
    labels : list[str]
        Human-readable class labels for visualization. Should match the number of classes
        in the data. If fewer labels are provided than classes detected, numeric labels
        will be generated automatically.
    threshold : float, optional
        Classification threshold for probability to class conversion.
        Default is 0.0.
    seaborn_style : str, optional
        Seaborn style name for plot aesthetics. Default is 'darkgrid'.
    matplotlib_style : str, optional
        Matplotlib style name for plot aesthetics. Default is 'fivethirtyeight'.

    Attributes
    ----------
    y_true : array-like
        Stored ground truth labels.
    y_preds_prob : array-like
        Stored prediction probabilities.
    y_preds : array-like
        Stored predicted labels.
    labels : list[str]
        Stored class labels.
    threshold : float
        Stored classification threshold.
    matplotlib_style : str
        Stored matplotlib style.

    Examples
    --------
    >>> from utils.metrics import ClassificationMetrics
    >>> # Binary classification
    >>> metrics = ClassificationMetrics(
    ...     y_true=[0, 1, 1, 0],
    ...     y_preds_prob=[0.1, 0.9, 0.8, 0.2],
    ...     y_preds=[0, 1, 1, 0],
    ...     labels=['Normal', 'Anomaly']
    ... )
    >>> cm = metrics.plot_confusion_matrix()
    >>> 
    >>> # Multiclass classification
    >>> metrics = ClassificationMetrics(
    ...     y_true=[0, 1, 2, 1, 0],
    ...     y_preds_prob=[0.1, 0.9, 0.8, 0.7, 0.2],
    ...     y_preds=[0, 1, 2, 1, 0],
    ...     labels=['Class A', 'Class B', 'Class C']
    ... )
    >>> cm = metrics.plot_confusion_matrix()
    
    Notes
    -----
    - For multiclass ROC and PR curves, the method uses one-vs-rest approach
      with the last class as the positive class
    - Confusion matrices automatically adapt to the number of classes detected
    - Warnings are logged when multiclass data is used with binary-oriented curves
    """

    def __init__(
        self,
        y_true: Any,
        y_preds_prob: Any,
        y_preds: Any,
        labels: List[str],
        threshold: float = 0.0,
        seaborn_style: str = 'darkgrid',
        matplotlib_style: str = 'fivethirtyeight'
    ) -> None:
        """Initialize ClassificationMetrics with predictions and labels."""
        try:
            self.y_true = y_true
            self.y_preds_prob = y_preds_prob
            self.y_preds = y_preds
            self.labels = labels
            self.threshold = threshold
            self.matplotlib_style = matplotlib_style
            
            sns.set_style(seaborn_style)
            logger.debug(
                f"Initialized ClassificationMetrics with {len(y_true)} samples, "
                f"threshold={threshold}"
            )
        except Exception as e:
            logger.error(f"Error initializing ClassificationMetrics: {e}")
            raise

    def _get_label_names(self, unique_labels: np.ndarray) -> List[str]:
        """
        Resolve human-readable label names for the given unique label values.

        - For binary problems (2 classes) returns ['Anomaly', 'Normal'] mapped
          to the actual label values (e.g. 0 -> 'Anomaly', 1 -> 'Normal').
        - For 4-class problems returns the domain-specific multiclass names.
        - If the user provided `self.labels` with matching length, those
          names are used (aligned to the order of `unique_labels`).
        - Otherwise falls back to numeric `Class {label}` names.
        """
        label_names: List[str] = []
        num_classes = len(unique_labels)

        # If user provided labels that match the number of classes, use them
        if len(self.labels) == num_classes:
            try:
                # Align provided labels to the order of unique_labels
                mapping = {}
                for idx, val in enumerate(unique_labels):
                    mapping[val] = self.labels[idx]
                return [mapping[val] for val in unique_labels]
            except Exception:
                return list(map(str, self.labels))

        # Predefined domain mappings
        binary_map = {0: 'Anomaly', 1: 'Normal'}
        multiclass_map = {
            0: 'Both anomaly',
            1: 'Point anomaly',
            2: 'Collective anomaly',
            3: 'Normal'
        }

        if num_classes == 2:
            for v in unique_labels:
                try:
                    label_names.append(binary_map.get(int(v), str(v)))
                except Exception:
                    label_names.append(str(v))
            return label_names

        if num_classes == 4:
            for v in unique_labels:
                try:
                    label_names.append(multiclass_map.get(int(v), str(v)))
                except Exception:
                    label_names.append(str(v))
            return label_names

        # Fallback: numeric class names
        for v in unique_labels:
            label_names.append(f"Class {v}")
        return label_names

    def plot_confusion_matrix(
        self,
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        cmap=plt.cm.Reds,
        colorbar: bool = True,
        label_rotation: int = 45
    ) -> np.ndarray:
        """
        Plot and compute the confusion matrix for binary or multiclass classification.

        This method generates a visual representation of the confusion matrix,
        showing the classification results for each class. For binary classification,
        it shows true positives, true negatives, false positives, and false negatives.
        For multiclass, it shows the full confusion matrix across all classes.
        The matrix can optionally be normalized to show proportions instead of raw counts.

        Parameters
        ----------
        normalize : bool, optional
            If True, normalize matrix to show proportions (0-1).
            If False, show raw counts. Default is False.
        title : str, optional
            Title for the confusion matrix plot. Default is 'Confusion Matrix'.
        cmap : matplotlib.colors.Colormap, optional
            Colormap for the matrix visualization. Default is plt.cm.Reds.
        colorbar : bool, optional
            Whether to display a color bar beside the matrix. Default is True.
        label_rotation : int, optional
            Rotation angle in degrees for x-axis labels. Default is 45.

        Returns
        -------
        np.ndarray
            Confusion matrix of shape (n_classes, n_classes). Values are
            normalized proportions if normalize=True, otherwise raw counts.
            The matrix automatically adapts to the number of classes detected.

        Notes
        -----
        The confusion matrix element [i, j] represents the number (or proportion)
        of samples with true label i that were predicted as label j.
        
        - Automatically detects the number of classes from the data
        - If more classes are detected than provided labels, numeric labels are generated
        - Works seamlessly with both binary and multiclass problems
        
        See Also
        --------
        sklearn.metrics.confusion_matrix : Underlying computation function
        """
        try:
            logger.debug(f"Plotting confusion matrix: normalize={normalize}")
            
            # Determine unique labels and create label list
            unique_labels = np.unique(np.concatenate([self.y_true, self.y_preds]))
            num_classes = len(unique_labels)
            
            # Resolve display label names (map numeric classes to human names)
            display_labels = self._get_label_names(unique_labels)
            
            # Compute confusion matrix with all unique labels
            cm = confusion_matrix(self.y_true, self.y_preds, labels=unique_labels)

            # Normalize matrix if requested
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, newaxis]
                title = title + ' (Normalized)'

            # Create plot
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            if colorbar:
                plt.colorbar()
            tick_marks = arange(len(display_labels))
            plt.xticks(tick_marks, display_labels, rotation=label_rotation)
            plt.yticks(tick_marks, display_labels)

            # Display values in matrix cells
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(
                    j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="#bbcdb4" if cm[i, j] > thresh else "#1b4c60",
                    fontsize=25
                )

            plt.tight_layout(pad=10)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            logger.debug(f"Confusion matrix plotted successfully")
            return cm
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            raise

    def plot_roc_curve(
        self,
        threshold: Optional[float] = None,
        plot_threshold: bool = True,
        linewidth: float = 2.0,
        c_roc_curve: str = '#1b4c60',
        c_random_guess: str = '#e34a6f',
        c_thresh_lines: str = '#bbcdb4',
        ls_roc_curve: str = '-',
        ls_thresh_lines: str = 'dashdot',
        ls_random_guess: str = '--',
        title: str = 'Receiver Operating Characteristic',
        loc_legend: str = 'lower right'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Compute and plot ROC curve with AUC score.

        This method generates a Receiver Operating Characteristic (ROC) curve,
        which plots the true positive rate against the false positive rate at
        various threshold settings. The area under the curve (AUC) provides a
        single metric for classifier performance.
        
        For multiclass problems, this method uses a one-vs-rest approach where
        the last class is treated as the positive class and all other classes
        are grouped as the negative class.

        Parameters
        ----------
        threshold : float, optional
            Classification threshold to visualize on the ROC curve.
            If None, uses self.threshold. Default is None.
        plot_threshold : bool, optional
            Whether to plot threshold lines and point on the ROC curve.
            Default is True.
        linewidth : float, optional
            Line width for plot elements. Default is 2.0.
        c_roc_curve : str, optional
            Color for the ROC curve line. Default is '#1b4c60'.
        c_random_guess : str, optional
            Color for the random guess diagonal line. Default is '#e34a6f'.
        c_thresh_lines : str, optional
            Color for threshold indicator lines. Default is '#bbcdb4'.
        ls_roc_curve : str, optional
            Linestyle for the ROC curve. Default is '-'.
        ls_thresh_lines : str, optional
            Linestyle for threshold lines. Default is 'dashdot'.
        ls_random_guess : str, optional
            Linestyle for random guess line. Default is '--'.
        title : str, optional
            Plot title. Default is 'Receiver Operating Characteristic'.
        loc_legend : str, optional
            Legend location ('lower right', 'upper left', etc.).
            Default is 'lower right'.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, float]
            A tuple containing:
            - fpr: False positive rates at each threshold
            - tpr: True positive rates at each threshold
            - thresh: Decision thresholds used for computing fpr and tpr
            - roc_auc: Area under the ROC curve

        Notes
        -----
        The ROC curve is a fundamental tool for binary classification evaluation.
        An AUC of 0.5 indicates random guessing, while 1.0 represents perfect
        classification.
        
        For multiclass problems (>2 classes), the method automatically converts
        to binary using one-vs-rest: the last class becomes the positive class,
        and all other classes are treated as negative. A warning is logged when
        this conversion occurs.
        
        References
        ----------
        .. [1] Fawcett, T. (2006). An introduction to ROC analysis.
               Pattern Recognition Letters, 27(8), 861-874.
        """
        try:
            logger.debug(f"Plotting ROC curve: threshold={threshold}, plot_threshold={plot_threshold}")
            
            if self.matplotlib_style is not None:
                plt.style.use(self.matplotlib_style)

            # Determine threshold to use
            t = threshold if threshold is not None else self.threshold

            # Check if we have binary or multiclass labels
            unique_labels = np.unique(self.y_true)
            if len(unique_labels) > 2:
                logger.warning(
                    f"Multiclass labels detected ({len(unique_labels)} classes). "
                    f"ROC curve requires binary labels. Converting to binary: "
                    f"class {unique_labels[-1]} vs rest."
                )
                # For multiclass, use one-vs-rest approach for the positive class
                # Assuming the last class (typically class 1 for "Normal") is the positive class
                y_true_binary = (self.y_true == unique_labels[-1]).astype(int)
                fpr, tpr, thresh = roc_curve(y_true_binary, self.y_preds_prob)
            else:
                # Binary classification
                fpr, tpr, thresh = roc_curve(self.y_true, self.y_preds_prob)
            
            roc_auc = auc(fpr, tpr)

            # Compute threshold coordinates
            idx_thresh = fpr[argmin(abs(thresh - t))]
            idy_thresh = tpr[argmin(abs(thresh - t))]

            # Plot random guess baseline
            plt.plot(
                [0, 1], [0, 1],
                color=c_random_guess,
                lw=linewidth,
                linestyle=ls_random_guess,
                label="Random guess"
            )
            
            # Plot threshold indicators if requested
            if plot_threshold:
                plt.axhline(y=idy_thresh, color=c_thresh_lines, linestyle=ls_thresh_lines, lw=linewidth)
                plt.axvline(x=idx_thresh, color=c_thresh_lines, linestyle=ls_thresh_lines, lw=linewidth)

            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                color=c_roc_curve,
                lw=linewidth,
                label=f'ROC curve (area = {roc_auc:.2f})',
                linestyle=ls_roc_curve
            )
            
            # Plot threshold point
            if plot_threshold:
                plt.plot(idx_thresh, idy_thresh, 'ro')

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc=loc_legend)

            logger.debug(f"ROC curve plotted successfully: AUC={roc_auc:.4f}")
            return fpr, tpr, thresh, roc_auc
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {e}")
            raise

    def plot_precision_recall_curve(
        self,
        threshold: Optional[float] = None,
        plot_threshold: bool = True,
        beta: float = 1.0,
        linewidth: float = 2.0,
        fscore_iso: Optional[List[float]] = None,
        iso_alpha: float = 0.7,
        c_pr_curve: str = '#1b4c60',
        c_thresh_lines: str = '#bbcdb4',
        c_f1_iso: str = '#e34a6f',
        c_thresh_point: str = 'red',
        ls_pr_curve: str = '-',
        ls_thresh: str = 'dashdot',
        ls_fscore_iso: str = ':',
        marker_pr_curve: Optional[str] = None,
        title: str = 'Precision and Recall Curve',
        loc_legend: str = 'lower left'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute and plot precision-recall curve with F-score iso-curves.

        This method generates a precision-recall curve showing the trade-off
        between precision and recall at various threshold settings. Optionally
        displays F-score iso-curves where F-score values are constant.
        
        For multiclass problems, this method uses a one-vs-rest approach where
        the last class is treated as the positive class and all other classes
        are grouped as the negative class.

        Parameters
        ----------
        threshold : float, optional
            Classification threshold to visualize on the curve.
            If None, uses self.threshold. Default is None.
        plot_threshold : bool, optional
            Whether to plot threshold indicators. Default is True.
        beta : float, optional
            Beta parameter for F-beta score calculation. beta=1 gives F1-score,
            beta=2 weighs recall higher, beta=0.5 weighs precision higher.
            Default is 1.0.
        linewidth : float, optional
            Line width for plot elements. Default is 2.0.
        fscore_iso : list[float], optional
            F-score values for iso-curves. If None, defaults to [0.2, 0.4, 0.6, 0.8].
            Set to empty list to disable iso-curves. Default is None.
        iso_alpha : float, optional
            Transparency of F-score iso-curves. Default is 0.7.
        c_pr_curve : str, optional
            Color for precision-recall curve. Default is '#1b4c60'.
        c_thresh_lines : str, optional
            Color for threshold indicator lines. Default is '#bbcdb4'.
        c_f1_iso : str, optional
            Color for F-score iso-curves. Default is '#e34a6f'.
        c_thresh_point : str, optional
            Color for threshold point marker. Default is 'red'.
        ls_pr_curve : str, optional
            Linestyle for precision-recall curve. Default is '-'.
        ls_thresh : str, optional
            Linestyle for threshold lines. Default is 'dashdot'.
        ls_fscore_iso : str, optional
            Linestyle for F-score iso-curves. Default is ':'.
        marker_pr_curve : str, optional
            Marker style for precision-recall curve. Default is None.
        title : str, optional
            Plot title. Default is 'Precision and Recall Curve'.
        loc_legend : str, optional
            Legend location. Default is 'lower left'.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - prec: Precision values at each threshold
            - recall: Recall values at each threshold
            - thresh: Decision thresholds

        Notes
        -----
        The precision-recall curve is particularly useful for imbalanced datasets
        where ROC curves may be overly optimistic. The area under the precision-recall
        curve (Average Precision) provides a single performance metric.
        
        For multiclass problems (>2 classes), the method automatically converts
        to binary using one-vs-rest: the last class becomes the positive class,
        and all other classes are treated as negative. A warning is logged when
        this conversion occurs.
        
        References
        ----------
        .. [1] Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is
               more informative than the ROC plot when evaluating binary classifiers
               on imbalanced datasets. PloS one, 10(3), e0118432.
        """
        try:
            logger.debug(
                f"Plotting precision-recall curve: threshold={threshold}, "
                f"beta={beta}, plot_threshold={plot_threshold}"
            )
            
            if self.matplotlib_style is not None:
                plt.style.use(self.matplotlib_style)
            
            # Set default F-score iso values and threshold
            if fscore_iso is None:
                fscore_iso = [0.2, 0.4, 0.6, 0.8]
            t = threshold if threshold is not None else self.threshold

            # Lists for legends
            lines, labels = [], []

            # Check if we have binary or multiclass labels
            unique_labels = np.unique(self.y_true)
            if len(unique_labels) > 2:
                logger.warning(
                    f"Multiclass labels detected ({len(unique_labels)} classes). "
                    f"Precision-Recall curve requires binary labels. Converting to binary: "
                    f"class {unique_labels[-1]} vs rest."
                )
                # For multiclass, use one-vs-rest approach for the positive class
                y_true_binary = (self.y_true == unique_labels[-1]).astype(int)
                prec, recall, thresh = precision_recall_curve(y_true_binary, self.y_preds_prob)
                pr_auc = average_precision_score(y_true_binary, self.y_preds_prob)
            else:
                # Binary classification
                prec, recall, thresh = precision_recall_curve(self.y_true, self.y_preds_prob)
                pr_auc = average_precision_score(self.y_true, self.y_preds_prob)
            
            # Compute threshold coordinates
            idx_thresh = recall[argmin(abs(thresh - t))]
            idy_thresh = prec[argmin(abs(thresh - t))]

            # Plot F-score iso-curves
            if len(fscore_iso) > 0:
                for f_score in fscore_iso:
                    x = linspace(0.005, 1, 100)
                    y = f_score * x / (beta ** 2 * x + x - beta ** 2 * f_score)
                    l, = plt.plot(
                        x[y >= 0], y[y >= 0],
                        color=c_f1_iso,
                        linestyle=ls_fscore_iso,
                        alpha=iso_alpha
                    )
                    plt.text(
                        s=f'F{str(beta)}={f_score:.1f}',
                        x=0.9,
                        y=y[-10] + 0.02,
                        alpha=iso_alpha
                    )
                lines.append(l)
                labels.append(f'F{str(beta)} curves')
                plt.ylim([-0.05, 1.05])

            # Plot threshold indicators
            if plot_threshold:
                plt.axhline(y=idy_thresh, color=c_thresh_lines, linestyle=ls_thresh, lw=linewidth)
                plt.axvline(x=idx_thresh, color=c_thresh_lines, linestyle=ls_thresh, lw=linewidth)

            # Plot precision-recall curve
            l, = plt.plot(
                recall, prec,
                color=c_pr_curve,
                lw=linewidth,
                linestyle=ls_pr_curve,
                marker=marker_pr_curve
            )
            lines.append(l)
            labels.append(f'PR curve (area = {pr_auc:.2f})')

            # Plot threshold point
            if plot_threshold:
                plt.plot(idx_thresh, idy_thresh, marker='o', color=c_thresh_point)

            # Configure plot
            plt.xlim([-0.05, 1.05])
            plt.legend(lines, labels, loc=loc_legend)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)

            logger.debug(f"Precision-recall curve plotted successfully: AP={pr_auc:.4f}")
            return prec, recall, thresh
        except Exception as e:
            logger.error(f"Error plotting precision-recall curve: {e}")
            raise

    def print_report(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate and print detailed classification reports.

        This method generates both standard and imbalanced classification reports,
        displaying metrics including precision, recall, F1-score, and support for
        each class. The imbalanced report includes additional metrics like
        geometric mean and index of balanced accuracy. Works for both binary
        and multiclass classification problems.

        Returns
        -------
        tuple[dict, dict]
            A tuple containing:
            - report: Standard classification report as dictionary
            - report_imb: Imbalanced classification report as dictionary

        Notes
        -----
        The standard report includes metrics for each class and overall averages.
        The imbalanced report is specifically designed for datasets with class
        imbalance and provides additional specialized metrics.
        
        - Automatically adapts to the number of classes in the data
        - If more classes are detected than provided labels, numeric labels are generated
        - Works seamlessly with both binary and multiclass problems

        See Also
        --------
        sklearn.metrics.classification_report : Standard classification metrics
        imblearn.metrics.classification_report_imbalanced : Imbalanced metrics
        """
        try:
            logger.debug("Generating classification reports")
            
            y_true_class = [int(y_t) for y_t in self.y_true]
            
            # Check if we have multiclass labels
            unique_labels = np.unique(self.y_true)
            num_classes = len(unique_labels)
            
            # Resolve human-readable label names for the report
            label_names = self._get_label_names(unique_labels)
            
            report = classification_report(
                self.y_true,
                self.y_preds,
                labels=unique_labels,
                target_names=label_names,
                output_dict=True
            )
            report_imb = classification_report_imbalanced(
                y_true_class,
                self.y_preds,
                labels=list(unique_labels),
                target_names=label_names,
                output_dict=True
            )

            print("                   ________________________ ")
            print("                  |  Classification Report |")
            print("                   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾ ")
            print(classification_report(self.y_true, self.y_preds, labels=unique_labels, target_names=label_names))

            logger.debug("Classification reports generated successfully")
            return report, report_imb
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            raise
