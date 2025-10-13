import os
import copy
import numpy as np
from .AdjointSolver import AdjointEvaluator
import time
import json


import os, json, copy
import numpy as np
from .AdjointSolver import AdjointEvaluator

class History:
    def __init__(self, source_file=None):
        self._data = []
        if source_file:
            self._read_in(source_file)

    @staticmethod
    def _coerce_json(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        # keep it simple: no special complex handling for now
        return x

    def _time(self):
        return time.datetime.now(time.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')

    def append(self, my_dict):
        new_dict = copy.deepcopy(my_dict)
        if "ts" not in new_dict:
            new_dict["ts"] = self._time
        self._data.append()

    def save(self, filename):
        """Append records to a .jsonl (one JSON object per line)."""
        path = filename if str(filename).endswith(".jsonl") else f"{filename}.jsonl"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:  # 'a' = append-only log
            for rec in self._data:
                out = {k: self._coerce_json(v) for k, v in rec.items()}
                f.write(json.dumps(out, separators=(",", ":")) + "\n")

    def _read_in(self, filename):
        """Load records from a .jsonl file (replace current _data)."""
        path = filename if str(filename).endswith(".jsonl") else f"{filename}.jsonl"
        self._data.clear()
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._data.append(json.loads(line))

class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.current_params = np.asarray(initial_params, float)
        self.lr = lr
        self.evaluator = evaluator
        self.history = History()

    def sweep(self, param_range: np.ndarray, perturbation_mag=None, verbose: bool = False):
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        for params in param_range:
            grad, loss = self.evaluator.evaluate(params, perturbation_mag, verbose=verbose)
            self.history.append({"params": np.asarray(params, float), "loss": float(loss), "grad": np.asarray(grad, float)})
        return self.history

    def gradient_descent(self, num_steps=50, perturbation_mag=None, verbose=False):
        if perturbation_mag is None:
            perturbation_mag = self.evaluator.param_perturbation[0]
        for k in range(num_steps):
            grad, loss = self.evaluator.evaluate(self.current_params, perturbation_mag, verbose=False)
            self.current_params = self.current_params - self.lr * grad
            self.history.append({"params": np.asarray(self.current_params, float), "loss": float(loss), "grad": np.asarray(grad, float)})
            if verbose:
                print(f"step {k}: loss={float(loss):.6e}, ||grad||={np.linalg.norm(grad):.6e}")
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