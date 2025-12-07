"""
validators.py — Input validation utilities for CoLog

This module provides comprehensive validation utilities for input parameters,
file paths, data structures, and configuration settings used throughout the
CoLog groundtruth extraction pipeline.

Classes
-------
InputValidator
    Static methods for validating various types of inputs including files,
    directories, DataFrames, pickle files, numeric ranges, and settings.
"""

import pickle
import logging
from pathlib import Path
from typing import Any, List, Dict
import pandas as pd
from .exceptions import ValidationError, FileNotFoundError, DataFormatError

logger = logging.getLogger(__name__)


class InputValidator(object):
    """Comprehensive input validation helper class."""
    
    @staticmethod
    def validate_file_exists(file_path: str, file_description: str = "File") -> None:
        """Validate that a file exists.
        
        Parameters
        ----------
        file_path : str
            Path to the file to validate.
        file_description : str, optional
            Description of the file for error messages.
        
        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_description} not found: {file_path}")
        if not path.is_file():
            raise ValidationError(f"{file_description} is not a file: {file_path}")
    
    @staticmethod
    def validate_directory_exists(dir_path: str, dir_description: str = "Directory") -> None:
        """Validate that a directory exists.
        
        Parameters
        ----------
        dir_path : str
            Path to the directory to validate.
        dir_description : str, optional
            Description of the directory for error messages.
        
        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"{dir_description} not found: {dir_path}")
        if not path.is_dir():
            raise ValidationError(f"{dir_description} is not a directory: {dir_path}")
    
    @staticmethod
    def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str], 
                                   dataset: str = "") -> None:
        """Validate that a DataFrame has required columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        required_columns : list[str]
            List of required column names.
        dataset : str, optional
            Dataset name for error context.
        
        Raises
        ------
        DataFormatError
            If required columns are missing.
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            prefix = f"[{dataset}] " if dataset else ""
            raise DataFormatError(
                f"{prefix}DataFrame missing required columns: {', '.join(missing)}. "
                f"Available columns: {', '.join(df.columns)}"
            )
    
    @staticmethod
    def validate_pickle_file(file_path: str, expected_type: type = None) -> None:
        """Validate that a pickle file can be loaded and optionally check type.
        
        Parameters
        ----------
        file_path : str
            Path to the pickle file.
        expected_type : type, optional
            Expected type of the pickled object.
        
        Raises
        ------
        ValidationError
            If file cannot be loaded or type doesn't match.
        """
        InputValidator.validate_file_exists(file_path, "Pickle file")
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            if expected_type is not None and not isinstance(obj, expected_type):
                raise ValidationError(
                    f"Pickle file {file_path} contains {type(obj).__name__}, "
                    f"expected {expected_type.__name__}"
                )
        except pickle.UnpicklingError as e:
            raise ValidationError(f"Failed to unpickle file {file_path}: {e}")
        except Exception as e:
            raise ValidationError(f"Error validating pickle file {file_path}: {e}")
    
    @staticmethod
    def validate_range(value: float, min_val: float, max_val: float, 
                      name: str, inclusive: bool = True) -> None:
        """Validate that a value is within a specified range.
        
        Parameters
        ----------
        value : float
            Value to validate.
        min_val : float
            Minimum allowed value.
        max_val : float
            Maximum allowed value.
        name : str
            Name of the parameter for error messages.
        inclusive : bool, optional
            Whether range is inclusive. Default is True.
        
        Raises
        ------
        ValidationError
            If value is out of range.
        """
        if inclusive:
            if not (min_val <= value <= max_val):
                raise ValidationError(
                    f"{name} must be between {min_val} and {max_val} (inclusive), got {value}"
                )
        else:
            if not (min_val < value < max_val):
                raise ValidationError(
                    f"{name} must be between {min_val} and {max_val} (exclusive), got {value}"
                )
    
    @staticmethod
    def validate_positive(value: float, name: str, allow_zero: bool = False) -> None:
        """Validate that a value is positive.
        
        Parameters
        ----------
        value : float
            Value to validate.
        name : str
            Name of the parameter for error messages.
        allow_zero : bool, optional
            Whether zero is allowed. Default is False.
        
        Raises
        ------
        ValidationError
            If value is not positive.
        """
        if allow_zero:
            if value < 0:
                raise ValidationError(f"{name} must be non-negative, got {value}")
        else:
            if value <= 0:
                raise ValidationError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def validate_in_list(value: Any, allowed_values: List[Any], name: str) -> None:
        """Validate that a value is in a list of allowed values.
        
        Parameters
        ----------
        value : Any
            Value to validate.
        allowed_values : list[Any]
            List of allowed values.
        name : str
            Name of the parameter for error messages.
        
        Raises
        ------
        ValidationError
            If value is not in the allowed list.
        """
        if value not in allowed_values:
            raise ValidationError(
                f"{name} must be one of {allowed_values}, got {value}"
            )
    
    @staticmethod
    def validate_settings_dict(settings: Dict[str, Any], required_keys: List[str], dataset: str = None) -> bool:
        """Validate that a settings dictionary contains all required keys.
        
        Parameters
        ----------
        settings : dict[str, Any]
            Settings dictionary to validate.
        required_keys : list[str]
            List of keys that must be present in settings.
        dataset : str, optional
            Dataset name for logging context. Default is None.
        
        Returns
        -------
        bool
            True if all required keys are present, False otherwise.
        
        Notes
        -----
        This method logs errors and prints messages for missing keys,
        making it suitable for user-facing validation.
        """
        missing = [k for k in required_keys if k not in settings]
        if missing:
            if dataset:
                logger.error('[%s] Missing required settings keys: %s', dataset, ', '.join(missing))
                print(f"[{dataset}] Error: Missing required settings keys: {', '.join(missing)}")
            else:
                logger.error('Missing required settings keys: %s', ', '.join(missing))
                print(f"Error: Missing required settings keys: {', '.join(missing)}")
            return False
        return True
