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

    def __str__(self):
        s = "N/A" if self.strength is None else np.array2string(self.strength, precision=8)
        loc = np.array2string(self.location, precision=8)
        dir_ = np.array2string(self.direction, precision=8)
        return f"Source(location={loc}, direction={dir_}, strength={s})"

