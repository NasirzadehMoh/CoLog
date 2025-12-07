"""
utils package for CoLog ground truth extraction

This package contains utility modules for the CoLog ground truth extraction system:
- cli: Command-line argument parsing
- constants: System-wide constants and configuration defaults
- settings: Dataset-specific settings management
- settings_provider: Settings provider interfaces and adapters
- exceptions: Custom exception classes
- validators: Input validation utilities
- dataclasses: Data structure definitions
"""

from .cli import parse_arguments
from . import constants
from .settings import Settings
from .settings_provider import ISettingsProvider, SettingsAdapter
from .validators import InputValidator
from .dataclasses import DatasetAttributes
from .exceptions import (
    GroundTruthError,
    ValidationError,
    FileNotFoundError,
    DataFormatError,
    ConfigurationError
)

__all__ = [
    'parse_arguments',
    'constants',
    'Settings',
    'ISettingsProvider',
    'SettingsAdapter',
    'InputValidator',
    'DatasetAttributes',
    'GroundTruthError',
    'ValidationError',
    'FileNotFoundError',
    'DataFormatError',
    'ConfigurationError'
]
