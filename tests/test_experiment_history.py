# test_experiment_history.py
import json
import os
import numpy as np
from adjoint_sim_sf import Experiment, Optimiser, AdjointEvaluator

def test_experiment_creates_dir(tmp_path):
    exp = Experiment(name="test_exp")
    # Update to use the new path - it creates results/test_exp
    assert os.path.exists(exp.dir)

def test_experiment_saves_results(tmp_path):
    exp = Experiment(name="test_exp")
    
    # Mock some results
    results = [
        {"loss": 1337, "grad": np.array([0.1, 0.2])},
        {"loss": 1337, "grad": np.array([0.05, 0.15])}
    ]
    
    exp.save_results(results, "test.jsonl")
    
    # Read it back
    with open(exp.path("test.jsonl")) as f:
        lines = f.readlines()
    
    assert len(lines) == 2
    data = json.loads(lines[0])
    assert data["loss"] == 1.5
    assert data["grad"] == [0.1, 0.2]

def test_experiment_saves_config(tmp_path):
    exp = Experiment(name="test_exp")
    config = {"freq_value": 8e9, "lr": 0.01}
    
    exp.save_config(config)
    
    with open(exp.path("config.json")) as f:
        loaded = json.load(f)
    
    assert loaded["freq_value"] == 8e9
    assert loaded["lr"] == 0.01

