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
    """
    @TODO: Units are not consistent between parameters, COMSOL simulation and QiskitMetal. 
    """
    def __init__(self, parametric_designer: ParametricDesign):
        if COMSOL_Model._engine is None:
            raise RuntimeError(
                "COMSOL engine not initialized. "
                "Call COMSOL_Model.init_engine() before running simulations."
            )
        self.parametric_designer = parametric_designer
        self.param_perturbation = np.array([1e-5])          # in mm
        self.freq_value = 8.0333e9
        self.fwd_source_strength = 1e3
        self.fwd_source_location = [300e-6, 300e-6, 100e-6]  # in meters
        self.adjoint_source_location = [0, 0, 100e-6]        # in meters
        self.sim_runner = SimulationRunner(self.freq_value)

        self.param_to_sim_scale = 1e-3

    def _fwd_calculation(self, design):
        return self.sim_runner.run_forward(design, self.fwd_source_location, self.fwd_source_strength)

    def _adjoint_calculation(self, design, adj_strength):
        return self.sim_runner.run_adjoint(design, self.adjoint_source_location, adj_strength)
    
    def _adjoint_strength(self, fwd_sparams, adjoint_location):
        # For the moment we are rescaling this 

        rescaling_factor = 1e-6

        raw_E_at_JJ = self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', np.array([adjoint_location]))
        adj_strength = (
            2 * np.real(raw_E_at_JJ[0, 1])
            / (2 * np.pi * self.freq_value * 1.256637e-6)
        )

        return adj_strength*rescaling_factor
    
    def save(self):
        self.sim_runner.save()


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

    def _calc_adjoint_forward_product(self, current_param, perturbation, fwd_sparams, adj_sparams):
        boundary_velocity_field, reference_coord, _ = self.parametric_designer.compute_boundary_velocity(current_param, perturbation)

        dr = 0.01 #TODO: extract this from the boundary field intelligently, or pass it from/to parametric designer -> polygon constructor

        running_sum = 0

        #TODO: Enforce non-2D
        r3 = 0.001

        #TODO: Vectorise this entire process and inner product for efficiency.
        
        #TODO: Do this cleverly with numpy

        flat_boundary_velocity_field =  [v for multipoly_velocities in boundary_velocity_field for v in multipoly_velocities[0]]
        flat_reference_coord = [coord_pair for multipoly_boundary_coord in reference_coord for coord_pair in multipoly_boundary_coord[0]]

        for v, (r1, r2) in zip(flat_boundary_velocity_field, flat_reference_coord):
            r1 = r1 * 1e-3
            r2 = r2 * 1e-3
            local_fwd_vec =     self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', [[r1, r2, r3]])
            local_adj_vec =     self.sim_runner.eval_field_at_pts(adj_sparams, 'E', [[r1, r2, r3]])  # extract E value of fwd_sparams at r1, r2
            
            print(local_fwd_vec.shape)
            print(local_adj_vec.shape)

            running_sum += v*(local_adj_vec.T@local_fwd_vec)*dr

        return running_sum 

    def evaluate(self, params: np.ndarray, verbose = False):
        """
        Run forward + adjoint sims for given params, return (loss, grad).
        Boundary velocity version
        """
        qk_design = self.parametric_designer.build_qk_design(params)

        # forward pass
        fwd_sparams = self._fwd_calculation(qk_design)
        # fwd_field_data = self.sim.eval_fields_over_mesh(fwd_sparams)
        raw_E_at_JJ = self.sim_runner.eval_field_at_pts(
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
        adj_E = self.sim_runner.eval_field_at_pts(
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

    def evaluate_old(self, params: np.ndarray, verbose = False):
        """Run forward + adjoint sims for given params, return (loss, grad)."""
        qk_design = self.parametric_designer.build_qk_design(params)

        # forward pass
        fwd_sparams = self._fwd_calculation(qk_design)
        # fwd_field_data = self.sim.eval_fields_over_mesh(fwd_sparams)
        raw_E_at_JJ = self.sim_runner.eval_field_at_pts(
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
        adj_E = self.sim_runner.eval_field_at_pts(
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
    
    def visualise(self, sims_params):
        field_data = self.sim_runner.eval_fields_over_mesh(sims_params)
        import matplotlib.pyplot as plt

        x,y,z,Ez = field_data['coords'][:,0], field_data['coords'][:,1], field_data['coords'][:,2], field_data['E'][:,2]

        plane_inds = (np.abs(z)<1e-6)
        plt.scatter(x[plane_inds], y[plane_inds], c=np.clip(np.abs(Ez[plane_inds]), 0,1e14))

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
    pass
    # import matplotlib.pyplot as plt
    # import numpy as np

    # if COMSOL_Model._engine is None:
    #     COMSOL_Model.init_engine()

    # sweep_vals = np.linspace(-0.05, 0.05, 21) + baseline_param

    # print("commencing sweep")
    # gradients = optimiser._compute_adjoint_gradient_sweep(sweep_vals)

    # COMSOL_Model.close_all_models()

    # for grad in gradients:
    #     print(grad)

    # dg = np.array(gradients)

    # plt.plot(np.linspace(-0.05, 0.05, 21), np.real(dg))
    # plt.plot(np.linspace(-0.05, 0.05, 21), np.imag(dg))
