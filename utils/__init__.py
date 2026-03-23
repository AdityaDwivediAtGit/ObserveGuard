"""
ObserveGuard Utils Module
Utilities for logging, energy tracking, and configuration
"""

from utils.codecarbon_wrapper import EnergyTracker, create_energy_tracker, MockEnergyTracker

__all__ = [
    'EnergyTracker',
    'create_energy_tracker',
    'MockEnergyTracker',
]
