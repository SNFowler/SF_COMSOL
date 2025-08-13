# AdjointSolver.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["PMIX_MCA_gds"]="hash"

# Import useful packages
import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict, open_docs

import matplotlib.pyplot as plt
import numpy as np

from SQDMetal.Comps.Junctions import JunctionDolan
import shapely
from shapely import difference
from shapely.geometry import MultiPolygon, LineString, Point
from SQDMetal.Comps.Polygons import PolyShapely, PolyRectangle
from SQDMetal.Comps.Joints import Joint

from SQDMetal.COMSOL.Model import COMSOL_Model
from SQDMetal.COMSOL.SimCapacitance import COMSOL_Simulation_CapMats
from SQDMetal.COMSOL.SimRFsParameter import COMSOL_Simulation_RFsParameters

from SQDMetal.Utilities.ShapelyEx import ShapelyEx
from typing import Iterable

from adjoint_sim_sf.ParametricDesign import ParametricDesign
from adjoint_sim_sf.Simulation import SimulationRunner  
    
class AdjointEvaluator:
    def __init__(self, parametric_designer: ParametricDesign, perturb: float = 1e-5):
        if COMSOL_Model._engine is None:
            raise RuntimeError(
                "COMSOL engine not initialized. "
                "Call COMSOL_Model.init_engine() before running simulations."
            )
        self.parametric_designer = parametric_designer
        self.param_perturbation = perturb
        self.freq_value = 8.0333e9
        self.fwd_source_location = [-25e-3, 2e-3, 100e-6]
        self.adjoint_source_location = [0, 0, 100e-6]
        self.sim = SimulationRunner(self.freq_value)

    # --- helpers ---
    def _fwd_calculation(self, design):
        design.rebuild()
        return self.sim.run_forward(design, self.fwd_source_location, 1.0)

    def _adjoint_calculation(self, design, adj_strength):
        design.rebuild()
        return self.sim.run_adjoint(design, self.adjoint_source_location, adj_strength)

    def _calc_Ap_x(self, fwd_field_data, poly_grad):
        def calc_grad(point):
            point = np.real(point)
            dist = np.sqrt(
                shapely.distance(
                    shapely.Point([point[0], point[1]]),
                    poly_grad
                ) ** 2 + point[2] ** 2
            )
            return np.exp(-0.5 * (dist / 0.002) ** 2)

        Ap_x = fwd_field_data['E'].copy()
        for j, mesh_coord in enumerate(fwd_field_data['coords']):
            Ap_x[j] *= calc_grad(mesh_coord)
        return Ap_x

    # --- main entry point ---
    def evaluate(self, params: np.ndarray, verbose: bool = False) -> tuple[float, np.ndarray]:
        """Run forward + adjoint sims for given params, return (loss, grad)."""
        design = self.parametric_designer.build_design(params)

        # forward pass
        fwd_sparams = self._fwd_calculation(design)
        fwd_field_data = self.sim.eval_fields_over_mesh(fwd_sparams)
        raw_E_at_JJ = self.sim.eval_field_at_pts(
            fwd_sparams, 'E', np.array([[0, 0, 0]])
        )

        # get A_p (shape derivative) from design
        poly_grad = self.parametric_designer.compute_Ap(
            params, self.param_perturbation
        )

        # adjoint pass
        adj_strength = (
            2 * np.real(raw_E_at_JJ[0, 1])
            / (2 * np.pi * self.freq_value * 1.256637e-6)
        )
        adj_sparams = self._adjoint_calculation(design, adj_strength)
        adj_E = self.sim.eval_field_at_pts(
            adj_sparams, 'E', fwd_field_data["coords"]
        )

        # compute gradient
        Ap_x = self._calc_Ap_x(fwd_field_data, poly_grad)
        grad_val = np.dot(adj_E.flatten(), Ap_x.flatten())

        if verbose:
            print(f"Grad = {np.real(grad_val)} + {np.imag(grad_val)}j")

        # placeholder loss
        loss_val = np.linalg.norm(raw_E_at_JJ)

        return loss_val, np.array([grad_val])

class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.params = np.asarray(initial_params, float)
        self.lr = float(lr)
        self.evaluator = evaluator
        self.history = []  # list of (params, loss)

    def step(self, verbose: bool = False):
        loss, grad = self.evaluator.evaluate(self.params, verbose=verbose)
        self.params -= self.lr * grad
        self.history.append((self.params.copy(), loss))
        return loss, grad

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    if COMSOL_Model._engine is None:
        COMSOL_Model.init_engine()

    # NOTE: your baseline_param/optimiser are assumed defined elsewhere.
    sweep_vals = np.linspace(-0.05, 0.05, 21) + baseline_param

    print("commencing sweep")
    gradients = optimiser._compute_adjoint_gradient_sweep(sweep_vals)

    COMSOL_Model.close_all_models()

    for grad in gradients:
        print(grad)

    dg = np.array(gradients)

    plt.plot(np.linspace(-0.05, 0.05, 21), np.real(dg))
    plt.plot(np.linspace(-0.05, 0.05, 21), np.imag(dg))
