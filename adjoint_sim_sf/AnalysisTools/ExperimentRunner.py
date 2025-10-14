"""
ExperimentRunner.py

Class for running experiments.

Handles saving history, high level management of runs.
"""
import json
import os
from datetime import datetime
import numpy as np

from adjoint_sim_sf import AdjointEvaluator, Optimiser

class Experiment:
    def __init__(self, name=None):
        self.name = name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
        self.dir = f"experiments/{self.name}"
        os.makedirs(self.dir, exist_ok=True)
    
    def path(self, filename):
        """Get full path for a file in this experiment's directory"""
        return os.path.join(self.dir, filename)
    
    def save_results(self, results, filename):
        """Save list of results to jsonl"""
        with open(self.path(filename), "w") as f:
            for result in results:
                f.write(json.dumps(result, default=self._json_convert) + "\n")
    
    def save_config(self, config, filename="config.json"):
        """Save config dict to json"""
        with open(self.path(filename), "w") as f:
            json.dump(config, f, indent=2, default=self._json_convert)
    
    @staticmethod
    def _json_convert(obj):
        """Handle numpy types for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} not JSON serializable")