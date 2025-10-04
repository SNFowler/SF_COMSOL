import os
import math
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
    @TODO: Velocity units wrong.
    """
    def __init__(self, parametric_designer: ParametricDesign):
        if COMSOL_Model._engine is None:
            raise RuntimeError(
                "COMSOL engine not initialized. "
                "Call COMSOL_Model.init_engine() before running simulations."
            )
        self.freq_value = 8.0333e9
        self.param_perturbation = np.array([1e-5])          # in mm
        self.fwd_source_strength = 1e-2
        self.fwd_source_locations = np.array([[300e-6, 300e-6, 100e-6]])  # in meters
        self.adjoint_source_locations = np.array([[0, 0, 100e-6]])        # in meters
        self.adjoint_rotation = math.pi/2 
        self.param_to_sim_scale = 1e-3

        self.parametric_designer = parametric_designer
        self.sim_runner = SimulationRunner(self.freq_value)

    def _fwd_calculation(self, design):
        return self.sim_runner.run_forward(design, self.fwd_source_locations, self.fwd_source_strength)

    def _adjoint_calculation(self, design, adj_strength):
        return self.sim_runner.run_adjoint(design, self.adjoint_source_locations, adj_strength)
    
    def _adjoint_strength(self, fwd_sparams: COMSOL_Simulation_RFsParameters, adjoint_locations: np.ndarray):
        """
        Evaluates the adjoint strength scalars at each source locations.

        Args:
            fwd_sparams: the output of the forward field simulation.
            adjoint_locations: vector of points to evaluate the field strength at
        """
        raw_E_at_JJ = self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', adjoint_locations)
        adj_strength = (
            2 * np.real(raw_E_at_JJ[0, 1])
            / (2 * np.pi * self.freq_value * 1.256637e-6)
        )

        return adj_strength
    
    def save(self):
        self.sim_runner.save()


    def _calc_Ap(self, current_param, fwd_field_data, perturbation_vec):
        """
        WIP: In the future will facilitate non-boundary velocity methods
        """
        raise NotImplementedError

    def _convert_unit(self, coord):
        # @TODO: Get this working for np.array and multidimensional coord.
        return coord * self.param_to_sim_scale


    def _calc_adjoint_forward_product(self, 
                                      boundary_velocity_field,
                                      reference_coord,
                                      fwd_sparams: COMSOL_Simulation_RFsParameters, 
                                      adj_sparams: COMSOL_Simulation_RFsParameters,
                                      adjoint_rotation: float = 0):
        

        dr = 0.005 # mm
         #TODO: extract this from the boundary field intelligently, or pass it from/to parametric designer -> polygon constructor

        running_sum = 0

        #TODO: Enforce non-2D
        r3 = 1e-6

        #TODO: Vectorise this entire process and inner product for efficiency.
        
        #TODO: Do this cleverly with numpy

        flat_boundary_velocity_field =  [v for multipoly_velocities in boundary_velocity_field for v in multipoly_velocities[0]]
        flat_reference_coord = [coord_pair for multipoly_boundary_coord in reference_coord for coord_pair in multipoly_boundary_coord[0]]

        phase = np.exp(1j * float(adjoint_rotation))

        for v, (r1, r2) in zip(flat_boundary_velocity_field, flat_reference_coord):
            v = self._convert_unit(v)
            r1 = self._convert_unit(r1)
            r2 = self._convert_unit(r2)

            # complete the forward and adjoint vectors. Note that the adjoint vector is actually the complex conjugate (but not the transpose) of the expected adjoint vector.

            local_fwd_vec =     self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', [[r1, r2, r3]])
            local_adj_vec_conj =     self.sim_runner.eval_field_at_pts(adj_sparams, 'E', [[r1, r2, r3]])  # extract E value of fwd_sparams at r1, r2
            
            local_adj_vec_conj = local_adj_vec_conj * phase

            # Taking the inner product. Note the vectors are intially in row form. 
            running_sum += v*(local_adj_vec_conj@local_fwd_vec.T)*dr

        return running_sum 

    def evaluate(self, params: np.ndarray, perturbation_vec, verbose = False):
        """
        Run forward + adjoint sims for given params, return (grad, loss). 
        grad is a complex number.
        Boundary velocity version
        """
        qk_design = self.parametric_designer.build_qk_design(params)

        # forward pass
        fwd_sparams = self._fwd_calculation(qk_design)
        # fwd_field_data = self.sim.eval_fields_over_mesh(fwd_sparams)
        
        adjoint_strength = self._adjoint_strength(
            fwd_sparams, self.adjoint_source_locations
        )

        adj_sparams = self._adjoint_calculation(qk_design, adjoint_strength)

        boundary_velocity_field, reference_coord, _ = self.parametric_designer.compute_boundary_velocity(params, perturbation_vec)

        inner_product = self._calc_adjoint_forward_product(
                boundary_velocity_field, reference_coord, fwd_sparams, adj_sparams,
                adjoint_rotation=self.adjoint_rotation)

        # For a real scalar objective, the gradient is the real part of the complex sensitivity
        grad_complex = -inner_product
        grad = 2.0 * np.real(grad_complex) 
        E_at_target =  self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', self.adjoint_source_locations)
      
        loss = (E_at_target) @ (np.conj(E_at_target).T)

        if verbose:
            print(f"Loss: {loss}Abs: {abs(grad)} Grad = {np.real(grad)} + {np.imag(grad)}j")

        return grad, loss
    
    def visualise(self, sims_params):
        field_data = self.sim_runner.eval_fields_over_mesh(sims_params)
        import matplotlib.pyplot as plt

        x,y,z,Ez = field_data['coords'][:,0], field_data['coords'][:,1], field_data['coords'][:,2], field_data['E'][:,2]

        plane_inds = (np.abs(z)<1e-6)
        plt.scatter(x[plane_inds], y[plane_inds], c=np.clip(np.abs(Ez[plane_inds]), 0,1e14))


    def sims(self, params, perturbation_vec):
        qk = self.parametric_designer.build_qk_design(params)
        fwd = self._fwd_calculation(qk)
        adj = self._adjoint_calculation(qk, self._adjoint_strength(fwd, self.adjoint_source_locations))
        loss = self.sim_runner.eval_field_at_pts(fwd, 'E', self.adjoint_source_locations)
        return fwd, adj, loss


class Optimiser:
    def __init__(self, initial_params: np.ndarray, lr: float, evaluator: AdjointEvaluator):
        self.params = np.asarray(initial_params, float)
        self.lr = float(lr)
        self.evaluator = evaluator
        self.history = []  # list of (params, loss)

    def step(self, verbose: bool = False):
        grad, loss = self.evaluator.evaluate(self.params, verbose=verbose)
        self.params -= self.lr * grad
        self.history.append((self.params.copy(), loss))
        return grad, loss
    
    def _make_param_range(self, center, width, num):
        return np.linspace(center - width, center + width, num)

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

    def sweep(self, center: float = 0.199, width: float = 0.04, num: int = 21,
              adjoint_rotation=None,
              perturbation_vec=None, verbose: bool = False, filename=None):
        if perturbation_vec is None:
            perturbation_vec = self.evaluator.param_perturbation

        param_range = self._make_param_range(center, width, num)
        losses, grads = [], []

        for x in param_range:
            p = [x]
            if adjoint_rotation: self.evaluator.adjoint_rotation = adjoint_rotation

            grad, loss = self.evaluator.evaluate(p, perturbation_vec, verbose=verbose)
            grads.append(grad)
            losses.append(loss)

        if filename:
            with self._open_file(filename) as f:
                for x, L, G in zip(param_range, losses, grads):
                    self._write_row(f, x, L, G)
        else: 
            print("no filename to save data")

        return param_range, losses, grads

    def sweep_reusing_fields(self, center=0.199, width=0.04, num=21,
                             adjoint_rotations=(0.0,),     
                             perturbation_vec=None, verbose=False, filename_base=None):
        if perturbation_vec is None:
            perturbation_vec = self.evaluator.param_perturbation

        param_range = self._make_param_range(center, width, num)

        for x in param_range:
            p = [x]
            fwd, adj, loss = self.evaluator.sims(p, perturbation_vec)
            boundary_velocity_field, reference_coord, _ = \
                self.evaluator.parametric_designer.compute_boundary_velocity(p, perturbation_vec)

            
            for adjoint_rotation in adjoint_rotations:
                grad = -self.evaluator._calc_adjoint_forward_product(
                    boundary_velocity_field, reference_coord,
                    fwd, adj, adjoint_rotation)
                if filename_base:
                    tag = f"ang={float(ang):.4f}rad"
                    with self._open_file(filename_base, tag=tag) as f:
                        self._write_row(f, x, loss, grad)

        if verbose:
            print(f"x={x:.6f} done")

        return param_range

    def gradient_descent(self, initial_param, lr=0.01, pertubation=None, num_steps=50, verbose=False):
        # Accept the existing (misspelled) argument name; map to the internal variable used elsewhere
        perturbation_vec = self.evaluator.param_perturbation if pertubation is None else pertubation

        # Initialise params and (optionally) learning rate for this run
        self.params = np.asarray(initial_param, dtype=float)
        local_lr = float(lr)

        grads = []
        losses = []

        for k in range(int(num_steps)):
            grad, loss = self.evaluator.evaluate(self.params, perturbation_vec, verbose=False)
           
            self.params -= local_lr * np.array([grad.imag])
            # Log minimal state
            self.history.append((self.params.copy(), loss))
            print(params)
            grads.append(grad)
            losses.append(loss)
            if verbose:
                G = complex(np.asarray(grad).ravel()[0])
                L = float(np.asarray(loss).ravel()[0])
                print(f"step={k:03d}  param={self.params.ravel()[0]:.10e}  loss={L:.10e}  grad={G.real:.10e}+{G.imag:.10e}j")

        return np.asarray(self.params, dtype=self.params.dtype), np.array(losses), np.array(grads)

        
        



if __name__ == "__main__":

    import matplotlib.pyplot as plt


    from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
    from SQDMetal.COMSOL.Model import COMSOL_Model

    if COMSOL_Model._engine is None:
            COMSOL_Model.init_engine()



    params = np.array([.119], dtype=float)
    perturbation_vec = np.array([0.01], dtype=float)

    # Build design and evaluator
    parametric_designer = SymmetricTransmonDesign()
    adjoint_evaluator = AdjointEvaluator(parametric_designer)
    design = parametric_designer.build_qk_design(params)
    optimiser = Optimiser(params, 0.01, adjoint_evaluator)

    param_range, losses, grads = optimiser.sweep(center=0.199, width=0.04, num=15, verbose=True)
    grads = np.array(grads)  # Convert list to array

    plt.plot(param_range, np.real(grads[:, 0, 0]))
    plt.plot(param_range, np.imag(grads[:, 0, 0]), 'y')
    plt.show()
