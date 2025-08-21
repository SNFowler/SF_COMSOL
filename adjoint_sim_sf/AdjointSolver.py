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

"""
@TODO: Naming confusion. "design" in this code refers to both the qiskit design and to an instance of ParametricDesign class.
"""
    
class AdjointEvaluator:
    def __init__(self, parametric_designer: ParametricDesign):
        if COMSOL_Model._engine is None:
            raise RuntimeError(
                "COMSOL engine not initialized. "
                "Call COMSOL_Model.init_engine() before running simulations."
            )
        self.parametric_designer = parametric_designer
        self.param_perturbation = np.array([1e-5])
        self.freq_value = 8.0333e9
        self.fwd_source_location = [-25e-3, 2e-3, 100e-6]
        self.adjoint_source_location = [0, 0, 100e-6]
        self.sim = SimulationRunner(self.freq_value)

    def _fwd_calculation(self, design):
        return self.sim.run_forward(design, self.fwd_source_location, 1.0)

    def _adjoint_calculation(self, design, adj_strength):
        return self.sim.run_adjoint(design, self.adjoint_source_location, adj_strength)

    def _calc_Ap(self, current_param, fwd_field_data, perturbation):
        """
        Compute the inner product between the adjoint and forward field, weighted by A_p.
        
        WIP
        """
        pass
        boundary_velocity_field, reference_coord, _ = self.parametric_designer.compute_boundary_velocity(current_param, perturbation)
        

        # dr = 1 #TODO: Comput this
        # for j, mesh_coord in enumerate(fwd_field_data['coords']):
        #     Ap_x[j] *= weight(mesh_coord)
        # return Ap_x

    def _calc_adjoint_forward_product(self, current_param, perturbation, fwd_sparams, bwd_sparams):
        boundary_velocity_field, reference_coord, _ = self.parametric_designer.compute_boundary_velocity(current_param, perturbation, dr)

        dr = 0.01 #TODO: extract this from the boundary field intelligently, or pass it from/to parametric designer -> polygon constructor

        running_sum = 0
        for v, (r1, r2) in zip(boundary_velocity_field, reference_coord):
            # extract E value of fwd_sparams at r1, r2
            # extract E value of bwd_sparams at r1, r2
            # running_sum += take the inner product, scale by v


    def evaluate(self, params: np.ndarray):
        """Run forward + adjoint sims for given params, return (loss, grad)."""
        qk_design = self.parametric_designer.build_qk_design(params)

        # forward pass
        fwd_sparams = self._fwd_calculation(qk_design)
        # fwd_field_data = self.sim.eval_fields_over_mesh(fwd_sparams)
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
        adj_sparams = self._adjoint_calculation(qk_design, adj_strength)
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

    sweep_vals = np.linspace(-0.05, 0.05, 21) + baseline_param

    print("commencing sweep")
    gradients = optimiser._compute_adjoint_gradient_sweep(sweep_vals)

    COMSOL_Model.close_all_models()

    for grad in gradients:
        print(grad)

    dg = np.array(gradients)

    plt.plot(np.linspace(-0.05, 0.05, 21), np.real(dg))
    plt.plot(np.linspace(-0.05, 0.05, 21), np.imag(dg))
