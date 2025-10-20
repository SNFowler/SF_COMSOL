import numpy as np
from typing import Iterator, Dict, Any
from .AdjointSolver import AdjointEvaluator

class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.current_params = np.asarray(initial_params, float)
        self.lr = lr
        self.evaluator = evaluator
    
    def sweep_generator(self, 
                       param_range: np.ndarray, 
                       perturbation_mag=None, 
                       verbose=False) -> Iterator[Dict[str, Any]]:
        """
        Generator that yields results one at a time.
        Allows caller to save incrementally.
        """
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        for i, params in enumerate(param_range):
            if verbose:
                print(f"Evaluating point {i}/{len(param_range)}: {params}")
            
            grad, loss = self.evaluator.evaluate(params, perturbation_mag, verbose=verbose)
            
            yield {
                "index": i,
                "params": np.asarray(params, float),
                "loss": float(loss),
                "grad": np.asarray(grad, float)
            }
    
    def sweep_multi_objective(self, 
                               param_range: np.ndarray, 
                               w_jj: float = 0.5, 
                               perturbation_mag=None, 
                               verbose=False) -> Iterator[Dict[str, Any]]:
        """
        Generator for multi-objective sweep.
        Yields results one at a time 
        """
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        assert 0 <= w_jj <= 1, "w_jj must be between 0 and 1"
        w_sa = 1 - w_jj
        
        for i, params in enumerate(param_range):
            if verbose:
                print(f"Evaluating point {i}/{len(param_range)}: {params}, w_jj={w_jj}")
            
            # This is where the actual computation happens
            result = self.evaluator.evaluate_multi_objective(
                params,
                perturbation_mag,
                w_jj=w_jj,
                w_sa=w_sa
            )
            
            # Add metadata
            result["index"] = i
            result["w_jj"] = float(w_jj)
            result["w_sa"] = float(w_sa)
            
            # yield = "pause here, return this value, resume when asked for next"
            yield result
    
    def gradient_descent_generator(self, 
                                   num_steps: int = 50, 
                                   perturbation_mag=None, 
                                   verbose=False) -> Iterator[Dict[str, Any]]:
        """
        Generator for gradient descent.
        Yields results one step at a time.
        """
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        for k in range(num_steps):
            grad, loss = self.evaluator.evaluate(self.current_params, perturbation_mag, verbose=False)
            
            result = {
                "step": k,
                "params": np.asarray(self.current_params, float),
                "loss": float(loss),
                "grad": np.asarray(grad, float),
                "grad_norm": float(np.linalg.norm(grad))
            }
            
            if verbose:
                print(f"step {k}: loss={float(loss):.6e}, ||grad||={result['grad_norm']:.6e}")
            
            # Update params AFTER yielding (so we save the params that produced this loss)
            self.current_params = self.current_params - self.lr * grad
            
            yield result
    
    # Keep old methods for backward compatibility
    def sweep(self, param_range: np.ndarray, perturbation_mag=None, verbose=False):
        """Batch version: collect all results before returning."""
        return list(self.sweep_generator(param_range, perturbation_mag, verbose))
    
    def sweep_jj_epr(self, param_range: np.ndarray, w_jj=0.5, perturbation_mag=None, verbose=False):
        """Batch version: collect all results before returning."""
        return list(self.sweep_multi_objective(param_range, w_jj, perturbation_mag, verbose))
    
    def gradient_descent(self, num_steps=50, perturbation_mag=None, verbose=False):
        """Batch version: collect all results before returning."""
        return list(self.gradient_descent_generator(num_steps, perturbation_mag, verbose))