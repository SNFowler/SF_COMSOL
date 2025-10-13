"""
Class for running experiments.

Handles saving history, high level management of runs.
"""
import json
from datetime import datetime

from adjoint_sim_sf import AdjointEvaluator, Optimiser

class Experiment:
    def __init__(self, exp_dir = None):
        if not exp_dir:
            exp_dir = f"C:\\Users\Experiment\OneDrive - The University of Queensland\\Desktop\SF_COMSOL\\notebooks\\{datetime.now().strftime("%Y%m%d-%H%M-%S")}"
        self.exp_dir = exp_dir

    def save_dict(self, my_dict, filename = None):
        if not filename:
            filename = self.exp_dr + f"\\{datetime.now().strftime("%Y%m%d-%H%M-%S")}.json"
        
        with open(filename, "w") as f:
            json.dump(my_dict, f, indent=2)
        return True

    def load_dict(self, filename):
        with open(filename, "r") as f:
            return json.load(f)