"""
Sources.py
Sources and Targets for EM Adjoint Optimisation
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class Source:
    # Infinitesimal Hertzian dipole source.
    location: np.ndarray  # meters [x, y, z]
    direction: np.ndarray  # unit vector
    strength: Optional[np.ndarray] = None # strength

    def __post_init__(self): 
        assert self.location.shape[0] == self.direction.shape[0]
        
    

