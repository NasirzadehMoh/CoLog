"""
exceptions.py — Custom exception classes for CoLog ground truth extraction

This module defines custom exception classes used throughout the CoLog ground truth
extraction system. These exceptions provide more specific error handling and better
error messages for various failure scenarios.
"""


class GroundTruthError(Exception):
    """Base exception for all groundtruth-related errors.
    
    This is the parent class for all custom exceptions in the ground truth
    extraction system. Catching this exception will catch all custom errors.
    """
    pass


class ValidationError(GroundTruthError):
    """Raised when input validation fails.
    
    This exception is raised when:
    - Input parameters are out of valid ranges
    - Required values are missing
    - Data types are incorrect
    - Constraints are violated
    """
    pass


class FileNotFoundError(GroundTruthError):
    """Raised when required files are not found.
    
    This exception is raised when:
    - Expected input files don't exist
    - Required directories are missing
    - File paths are invalid
    """
    pass


class DataFormatError(GroundTruthError):
    """Raised when data doesn't have expected format or columns.
    
    This exception is raised when:
    - DataFrame is missing required columns
    - Data structures have unexpected formats
    - File contents don't match expected schema
    """
    pass


class ConfigurationError(GroundTruthError):
    """Raised when configuration is invalid or incomplete.
    
    This exception is raised when:
    - Settings are missing required keys
    - Configuration values are invalid
    - System setup is incomplete
    """
    pass
