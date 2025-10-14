# test_experiment_history.py
import json
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
        {"loss": 1.5, "grad": np.array([0.1, 0.2])},
        {"loss": 1.3, "grad": np.array([0.05, 0.15])}
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

def test_optimiser_returns_results():
    # You'll need to mock or use a real evaluator here
    # Simplified example assuming you have a mock evaluator
    
    initial_params = np.array([0.2, 0.3])
    lr = 0.01
    # evaluator = MockEvaluator()  # You'd create this
    # opt = Optimiser(initial_params, lr, evaluator)
    
    # results = opt.gradient_descent(num_steps=5)
    
    # assert len(results) == 5
    # assert "loss" in results[0]
    # assert "grad" in results[0]
    # assert "params" in results[0]
    pass  # Remove once you have a mock evaluator