import os
import numpy as np
from .AdjointSolver import AdjointEvaluator


class History:
    def __init__(self):
        self._data = []  # list of (params, loss, grad)
    
    def append(self, params, loss, grad):
        self._data.append((params.copy(), loss, grad))
    
    def save(self, filename):
        """Save all history data to file."""
        with self._open_file(filename) as f:
            for params, loss, grad in self._data:
                x = params[0]  # Assumes single parameter
                self._write_row(f, x, loss, grad)
    
    def _open_file(self, filename, tag=None):
        fn = filename if str(filename).endswith(".dat") else f"{filename}.dat"
        if tag:
            fn = os.path.join(filename, f"{tag}.dat")
        d = os.path.dirname(fn)
        if d:
            os.makedirs(d, exist_ok=True)
        exists = os.path.exists(fn)
        f = open(fn, "a" if exists else "w")
        if not exists:
            f.write("param\tloss\treal_grad\timag_grad\tabs_grad\n")
        return f
    
    def _write_row(self, f, x, loss, grad):
        L = float(np.asarray(loss).ravel()[0])
        G = complex(np.asarray(grad).ravel()[0])
        f.write(f"{x:.10e}\t{L:.10e}\t{G.real:.10e}\t{G.imag:.10e}\t{abs(G):.10e}\n")

import os
import numpy as np
from .AdjointSolver import AdjointEvaluator


class History:
    def __init__(self):
        self.data = []  # list of dict: {'params': array, 'loss': float, 'grad': array}
    
    def append(self, params, loss, grad):
        self.data.append({
            'params': params.copy(),
            'loss': float(np.real(loss)),
            'grad': grad.copy() if grad is not None else None
        })
    
    def save(self, filename):
        """Save history to numpy file."""
        np.savez(filename, 
                 params=np.array([d['params'] for d in self.data]),
                 losses=np.array([d['loss'] for d in self.data]),
                 grads=np.array([d['grad'] for d in self.data if d['grad'] is not None]))


class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.current_params = np.asarray(initial_params, float)
        self.lr = lr
        self.evaluator = evaluator
        self.history = History()

    def sweep(self, param_range: np.ndarray, perturbation_mag=None, verbose: bool = False):
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        
        for params in param_range:  # params can be 1D or multi-dimensional
            grad_vec, loss = self.evaluator.evaluate(params, perturbation_mag, verbose=verbose)
            self.history.append(params, loss, grad_vec)
        
        return self.history

    def gradient_descent(self, num_steps=50, perturbation_mag=None, verbose=False):
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]

        for k in range(num_steps):
            grad_vec, loss = self.evaluator.evaluate(self.current_params, perturbation_mag, verbose=False)
            self.current_params -= self.lr * grad_vec
            self.history.append(self.current_params, loss, grad_vec)
            
            if verbose:
                print(f"step {k}: loss={float(loss):.6e}, ||grad||={np.linalg.norm(grad_vec):.6e}")

        return self.current_params, self.history

        # Unused 
        # def sweep_reusing_fields(self, center=0.199, width=0.04, num=21,
        #                      angles=(0.0,),     
        #                      perturbation=None, verbose=False, filename_base=None):
        # if perturbation is None:
        #     perturbation = self.evaluator.param_perturbation

        # param_range = self._make_param_range(center, width, num)

        # for x in param_range:
        #     p = [x]
        #     fwd, adj, loss = self.evaluator.sims(p, perturbation)
        #     boundary_velocity_field, reference_coord, _ = \
        #         self.evaluator.parametric_designer.compute_boundary_velocity(p, perturbation)

            
        #     for ang in angles:
        #         grad = -self.evaluator._calc_adjoint_forward_product(
        #             boundary_velocity_field, reference_coord,
        #             fwd, adj, ang)
        #         if filename_base:
        #             tag = f"ang={float(ang):.4f}rad"
        #             with self._open_file(filename_base, tag=tag) as f:
        #                 self._write_row(f, x, loss, grad)

        # if verbose:
        #     print(f"x={x:.6f} done")

        # return param_range