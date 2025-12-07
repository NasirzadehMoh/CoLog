"""
groundtruth_detector.py — Auto-detection utility for groundtruth configurations

This module provides functionality to automatically detect available groundtruth
configurations for a dataset without requiring manual specification of train-ratio,
sequence-type, and window-size parameters.

Key Features:
    - Scans dataset directories for available groundtruth configurations
    - Detects sequence types (context/background) and window sizes
    - Finds available train/valid/test split ratios
    - Provides intelligent defaults when multiple options exist
    - Validates detected configurations

Usage:
    from train.utils.groundtruth_detector import detect_groundtruth_config
    
    config = detect_groundtruth_config('casper-rw')
    # Returns: {'sequence_type': 'context', 'window_size': 1, 'train_ratio': 0.6}

Dependencies: pathlib, logging
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import constants

logger = logging.getLogger(__name__)


def detect_groundtruth_config(
    dataset_name: str,
    dataset_dir: Optional[Path] = None,
    prefer_sequence_type: Optional[str] = None,
    prefer_window_size: Optional[int] = None,
    prefer_train_ratio: Optional[float] = None,
    prefer_resample_method: Optional[str] = None
) -> Dict[str, any]:
    """
    Automatically detect groundtruth configuration for a dataset.
    
    This function scans the dataset's groundtruth directory to find available
    configurations and returns the best match based on preferences or defaults.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'casper-rw', 'hadoop', 'spark')
    dataset_dir : Path, optional
        Path to datasets directory. If None, uses default 'datasets/'
    prefer_sequence_type : str, optional
        Preferred sequence type ('context' or 'background'). If None, auto-selects.
    prefer_window_size : int, optional
        Preferred window size. If None, auto-selects.
    prefer_train_ratio : float, optional
        Preferred train ratio. If None, auto-selects.
    prefer_resample_method : str, optional
        Preferred resample method (e.g., 'randomundersampler', 'tomeklinks'). If None, auto-selects.
    
    Returns
    -------
    dict
        Configuration dictionary with keys:
        - 'sequence_type': str - Detected or preferred sequence type
        - 'window_size': int - Detected or preferred window size
        - 'train_ratio': float or None - Detected or preferred train ratio
        - 'resample_method': str or None - Detected or preferred resample method
        - 'available_configs': list - All available configurations found
    
    Raises
    ------
    FileNotFoundError
        If no groundtruth configurations are found for the dataset
    ValueError
        If preferred configuration is not available
    
    Examples
    --------
    >>> config = detect_groundtruth_config('casper-rw')
    >>> print(config)
    {'sequence_type': 'context', 'window_size': 1, 'train_ratio': 0.6, ...}
    
    >>> config = detect_groundtruth_config('casper-rw', prefer_window_size=2)
    >>> print(config['window_size'])
    2
    """
    if dataset_dir is None:
        dataset_dir = Path('datasets')
    
    groundtruth_dir = dataset_dir / dataset_name / constants.DIR_GROUNDTRUTH
    
    if not groundtruth_dir.exists():
        raise FileNotFoundError(
            f"Groundtruth directory not found: {groundtruth_dir}"
        )
    
    # Scan for available configurations
    available_configs = scan_groundtruth_configs(groundtruth_dir, dataset_name)
    
    if not available_configs:
        raise FileNotFoundError(
            f"No groundtruth configurations found in {groundtruth_dir}"
        )
    
    logger.info(f"Found {len(available_configs)} groundtruth configurations for {dataset_name}")
    
    # Select best configuration based on preferences
    selected_config = select_best_config(
        available_configs,
        prefer_sequence_type,
        prefer_window_size,
        prefer_train_ratio,
        prefer_resample_method
    )
    
    resample_info = f", resample_method={selected_config['resample_method']}" if selected_config.get('resample_method') else ""
    logger.info(
        f"Selected config: {selected_config['sequence_type']}_"
        f"{selected_config['window_size']}, train_ratio={selected_config['train_ratio']}"
        f"{resample_info}"
    )
    
    return selected_config


def scan_groundtruth_configs(
    groundtruth_dir: Path,
    dataset_name: str
) -> List[Dict[str, any]]:
    """
    Scan groundtruth directory for all available configurations.
    
    Parameters
    ----------
    groundtruth_dir : Path
        Path to the groundtruth directory
    dataset_name : str
        Name of the dataset
    
    Returns
    -------
    list of dict
        List of configuration dictionaries, each containing:
        - 'sequence_type': str
        - 'window_size': int
        - 'train_ratio': float or None
        - 'resample_method': str or None
        - 'path': Path
        - 'has_splits': bool
        - 'has_resampled': bool
    """
    configs = []
    
    # Scan sequence type directories (e.g., context_1, background_2)
    for seq_dir in groundtruth_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        
        # Parse sequence_type and window_size from directory name
        parts = seq_dir.name.split('_')
        if len(parts) < 2:
            logger.warning(f"Skipping invalid directory name: {seq_dir.name}")
            continue
        
        sequence_type = parts[0]
        try:
            window_size = int(parts[1])
        except ValueError:
            logger.warning(f"Invalid window size in directory: {seq_dir.name}")
            continue
        
        # Check for train_valid_test splits
        has_splits = False
        train_ratios = []
        
        for subdir in seq_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            if subdir.name.startswith(constants.DIR_TRAIN_VALID_TEST_PREFIX):
                has_splits = True
                # Extract train ratio (e.g., train_valid_test_0.6 -> 0.6)
                try:
                    ratio_str = subdir.name.split('_')[-1]
                    train_ratio = float(ratio_str)
                    train_ratios.append(train_ratio)
                    
                    # Check for resampled groundtruth methods
                    resampled_dir = subdir / constants.DIR_RESAMPLED
                    resample_methods = []
                    
                    if resampled_dir.exists() and resampled_dir.is_dir():
                        for method_dir in resampled_dir.iterdir():
                            if method_dir.is_dir():
                                resample_methods.append(method_dir.name)
                    
                    # Add configuration without resampling
                    configs.append({
                        'sequence_type': sequence_type,
                        'window_size': window_size,
                        'train_ratio': train_ratio,
                        'resample_method': None,
                        'path': subdir,
                        'has_splits': True,
                        'has_resampled': len(resample_methods) > 0
                    })
                    
                    # Add configurations for each resample method
                    for method in resample_methods:
                        configs.append({
                            'sequence_type': sequence_type,
                            'window_size': window_size,
                            'train_ratio': train_ratio,
                            'resample_method': method,
                            'path': resampled_dir / method,
                            'has_splits': True,
                            'has_resampled': True
                        })
                except (ValueError, IndexError):
                    logger.warning(f"Invalid train ratio in directory: {subdir.name}")
        
        # If no splits found but directory exists, add it with train_ratio=None
        # Note: We skip 'all' directories when splits exist, as they're just backups
        if not has_splits:
            configs.append({
                'sequence_type': sequence_type,
                'window_size': window_size,
                'train_ratio': None,
                'resample_method': None,
                'path': seq_dir,
                'has_splits': False,
                'has_resampled': False
            })
    
    return configs


def select_best_config(
    available_configs: List[Dict[str, any]],
    prefer_sequence_type: Optional[str] = None,
    prefer_window_size: Optional[int] = None,
    prefer_train_ratio: Optional[float] = None,
    prefer_resample_method: Optional[str] = None
) -> Dict[str, any]:
    """
    Select the best configuration from available options.
    
    If preferences don't narrow down to a single configuration, prompts the user
    to select from the available options.
    
    Selection priority:
    1. Exact match of all preferences (if provided)
    2. Match sequence_type and window_size preferences
    3. Prompt user to select from remaining candidates
    
    Parameters
    ----------
    available_configs : list of dict
        List of available configurations
    prefer_sequence_type : str, optional
        Preferred sequence type
    prefer_window_size : int, optional
        Preferred window size
    prefer_train_ratio : float, optional
        Preferred train ratio
    prefer_resample_method : str, optional
        Preferred resample method
    
    Returns
    -------
    dict
        Selected configuration with 'available_configs' list added
    
    Raises
    ------
    ValueError
        If preferred configuration is not available or invalid selection
    """
    if not available_configs:
        raise ValueError("No configurations available to select from")
    
    candidates = available_configs.copy()
    
    # Filter by sequence type preference
    if prefer_sequence_type:
        filtered = [c for c in candidates if c['sequence_type'] == prefer_sequence_type]
        if not filtered:
            raise ValueError(
                f"Preferred sequence_type '{prefer_sequence_type}' not found. "
                f"Available: {set(c['sequence_type'] for c in candidates)}"
            )
        candidates = filtered
    
    # Filter by window size preference
    if prefer_window_size is not None:
        filtered = [c for c in candidates if c['window_size'] == prefer_window_size]
        if not filtered:
            raise ValueError(
                f"Preferred window_size {prefer_window_size} not found. "
                f"Available: {sorted(set(c['window_size'] for c in candidates))}"
            )
        candidates = filtered
    
    # Filter by train ratio preference
    if prefer_train_ratio is not None:
        filtered = [c for c in candidates if c['train_ratio'] == prefer_train_ratio]
        if not filtered:
            raise ValueError(
                f"Preferred train_ratio {prefer_train_ratio} not found. "
                f"Available: {sorted(set(c['train_ratio'] for c in candidates if c['train_ratio'] is not None))}"
            )
        candidates = filtered
    
    # Filter by resample method preference
    if prefer_resample_method is not None:
        filtered = [c for c in candidates if c.get('resample_method') == prefer_resample_method]
        if not filtered:
            available_methods = sorted(set(c.get('resample_method') for c in candidates if c.get('resample_method') is not None))
            raise ValueError(
                f"Preferred resample_method '{prefer_resample_method}' not found. "
                f"Available: {available_methods}"
            )
        candidates = filtered
    
    # If we still have multiple candidates, prompt user to select
    if len(candidates) > 1:
        print("\nMultiple groundtruth configurations found:")
        print("-" * 120)
        for idx, config in enumerate(candidates, 1):
            train_ratio_str = f"{config['train_ratio']:.1f}" if config['train_ratio'] is not None else "None"
            resample_str = config.get('resample_method') if config.get('resample_method') else "Not resampled"
            print(f"  [{idx}] Sequence Type: {config['sequence_type']:<12} "
                  f"Window Size: {config['window_size']:<3} "
                  f"Train Ratio: {train_ratio_str:<6} "
                  f"Resample Method: {resample_str:<20}")
            # print(f"       Path: {config['path']}")
            # print()
        print("-" * 120)
        
        while True:
            try:
                choice = input(f"Please select a configuration [1-{len(candidates)}]: ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(candidates):
                    selected = candidates[choice_idx]
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(candidates)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nConfiguration selection cancelled.")
                raise ValueError("User cancelled configuration selection")
    else:
        selected = candidates[0]
    
    selected['available_configs'] = available_configs

    return selected


def get_groundtruth_path(
    dataset_name: str,
    sequence_type: str,
    window_size: int,
    train_ratio: Optional[float] = None,
    resample_method: Optional[str] = None,
    dataset_dir: Optional[Path] = None
) -> Path:
    """
    Construct groundtruth path from configuration parameters.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    sequence_type : str
        Sequence type ('context' or 'background')
    window_size : int
        Window size
    train_ratio : float, optional
        Train ratio (None for datasets without splits)
    resample_method : str, optional
        Resample method (None for non-resampled data)
    dataset_dir : Path, optional
        Path to datasets directory
    
    Returns
    -------
    Path
        Full path to groundtruth directory
    """
    if dataset_dir is None:
        dataset_dir = Path('datasets')
    
    dataset_config = f"{sequence_type}_{window_size}"
    
    if train_ratio is not None:
        # Datasets with train/valid/test splits
        base_path = (dataset_dir / dataset_name / constants.DIR_GROUNDTRUTH / 
                     dataset_config / f'{constants.DIR_TRAIN_VALID_TEST_PREFIX}{train_ratio}')
        
        if resample_method is not None:
            # Resampled groundtruth data
            return base_path / constants.DIR_RESAMPLED / resample_method
        else:
            # Non-resampled groundtruth data
            return base_path
    else:
        # Datasets without splits
        return dataset_dir / dataset_name / constants.DIR_GROUNDTRUTH / dataset_config