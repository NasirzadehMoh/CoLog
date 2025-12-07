"""
settings_provider.py — Settings provider interface and adapter for CoLog

This module provides an abstraction layer for accessing dataset settings,
enabling dependency injection and facilitating testing by allowing mock
settings providers to be used in place of the concrete Settings class.

Classes
-------
ISettingsProvider
    Abstract interface defining the contract for settings providers.
SettingsAdapter
    Concrete adapter that wraps the Settings class to implement the
    ISettingsProvider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .settings import Settings


class ISettingsProvider(ABC):
    """Interface for settings providers to enable dependency injection and testing.
    
    This abstract base class defines the contract that all settings providers
    must implement. It enables dependency injection by allowing different
    implementations to be passed to classes that need settings, making the
    code more testable and flexible.
    """
    
    @abstractmethod
    def get_settings(self) -> Dict[str, Any]:
        """Get settings dictionary for the dataset.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing all settings for the dataset, including
            paths, parser configurations, and other dataset-specific parameters.
        """
        pass


class SettingsAdapter(ISettingsProvider):
    """Adapter that wraps the Settings class to implement ISettingsProvider.
    
    This adapter provides a bridge between the concrete Settings class and
    the ISettingsProvider interface, allowing the Settings class to be used
    wherever an ISettingsProvider is expected.
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to load settings for (e.g., 'hadoop', 'bgl').
    
    Examples
    --------
    >>> adapter = SettingsAdapter('hadoop')
    >>> settings = adapter.get_settings()
    >>> print(settings['log_format'])
    """
    
    def __init__(self, dataset: str) -> None:
        """Initialize the adapter with a dataset name.
        
        Parameters
        ----------
        dataset : str
            Name of the dataset to load settings for.
        """
        self._settings = Settings(dataset)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get settings dictionary for the dataset.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing all settings for the dataset.
        """
        return self._settings.get_settings()
