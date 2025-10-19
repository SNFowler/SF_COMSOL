import os, json, copy, time
import numpy as np
from .AdjointSolver import AdjointEvaluator
class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.current_params = np.asarray(initial_params, float)
        self.lr = lr
        self.evaluator = evaluator
    
    def sweep(self, param_range: np.ndarray, perturbation_mag=None, verbose=False):
        """Evaluate at each point in param_range, return list of results"""
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        results = []
        for params in param_range:
            grad, loss = self.evaluator.evaluate(params, perturbation_mag, verbose=verbose)
            results.append({
                "params": np.asarray(params, float),
                "loss": float(loss),
                "grad": np.asarray(grad, float)
            })
        return results
    
    def sweep_jj_epr(self, param_range: np.ndarray, w_jj = 0.5, perturbation_mag=None, verbose=False):
        """Evaluate at each point in param_range, return list of results"""
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        assert w_jj >= 0 and w_jj <= 1
        w_sa = 1 - w_jj

        results = []
        for params in param_range:
            current_results = self.evaluator.evaluate_multi_objective(
                                params,
                                perturbation_mag,
                                w_jj=w_jj,
                                w_sa=w_sa)
            
            results.append(current_results)
        return results
    
    def gradient_descent(self, num_steps=50, perturbation_mag=None, verbose=False):
        """Run gradient descent, return list of results at each step"""
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        results = []
        for k in range(num_steps):
            grad, loss = self.evaluator.evaluate(self.current_params, perturbation_mag, verbose=False)
            self.current_params = self.current_params - self.lr * grad
            
            results.append({
                "step": k,
                "params": np.asarray(self.current_params, float),
                "loss": float(loss),
                "grad": np.asarray(grad, float)
            })
            
            if verbose:
                print(f"step {k}: loss={float(loss):.6e}, ||grad||={np.linalg.norm(grad):.6e}")
        
        return results