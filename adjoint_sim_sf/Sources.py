"""
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
    strength: float
    

