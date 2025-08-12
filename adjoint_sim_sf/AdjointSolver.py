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

from adjoint_sim_sf import DesignBuilder, PolygonConstructor
from adjoint_sim_sf.Simulation import SimulationRunner  


class AdjointOptimiser:
    def __init__(self, initial_parameters: np.array, lr: float,
                 qiskit_builder: DesignBuilder, shapely_constructor: PolygonConstructor):
        if COMSOL_Model._engine is None:
            raise RuntimeError("COMSOL engine not initialized. Call COMSOL_Model.init_engine() before running simulations.")

        self.qiskit_builder = qiskit_builder
        self.shapely_constructor = shapely_constructor
        self.design = None
        self.current_params = initial_parameters
        self.lr = lr

        self.fwd_field_sParams = None
        self.adjoint_field_sParams = None
        self.fwd_E_data = None
        self.adjoint_E_data = None

        self.current_Ap = None
        self.param_perturbation = 1e-5
        self.grad = None

        self.freq_value = 8.0333e9
        self.fwd_source_location = [-25e-3, 2e-3, 100e-6]
        self.adjoint_source_location = [0, 0, 100e-6]

        self.sim = SimulationRunner(self.freq_value)      # ← NEW

    # ---- helper wrappers now delegate to the runner ----
    def _get_field_at_junction(self, sim):
        return self.sim.eval_field_at_pts(sim, 'E', np.array([[0, 0, 0]]))

    def _get_field_over_mesh(self, sim):
        return self.sim.eval_fields_over_mesh(sim)

    # ---- core steps ----
    def _fwd_calculation(self):
        assert self.design
        self.design.rebuild()
        self.fwd_field_sParams = self.sim.run_forward(self.design, self.fwd_source_location, 1.0)

    def _adjoint_calculation(self):
        assert self.design and self.fwd_field_sParams
        raw_E = self._get_field_at_junction(self.fwd_field_sParams)
        adj_strength = 2 * np.real(raw_E[0, 1]) / (2 * np.pi * self.freq_value * 1.256637e-6)
        self.design.rebuild()
        self.adjoint_field_sParams = self.sim.run_adjoint(self.design, self.adjoint_source_location, adj_strength)

    def _update_qiskit_design(self):
        shapely_components = self.shapely_constructor.make_polygons(self.current_params)
        print(shapely_components)
        print(len(shapely_components))
        self.design = self.qiskit_builder.get_design(shapely_components)

    def _extract_E_fields(self):
        assert self.fwd_field_sParams is not None, "Forward simulation must be run first."
        self.fwd_field_data = self._get_field_over_mesh(self.fwd_field_sParams)

        assert self.adjoint_field_sParams is not None, "Adjoint simulation must be run first."
        self.adjoint_E_data = {
            "coords": self.fwd_field_data["coords"],
            "E": self.sim.eval_field_at_pts(self.adjoint_field_sParams, 'E', self.fwd_field_data["coords"])
        }

    def _compute_Ap(self) -> list:
        base_shape = self.shapely_constructor.make_polygons(self.current_params)
        differences = []
        for i in range(len(self.current_params)):
            perturbed_params = np.copy(self.current_params)
            perturbed_params[i] += self.param_perturbation
            perturbed_shape = self.shapely_constructor.make_polygons(perturbed_params)
            difference = shapely.difference(shapely.unary_union(base_shape), shapely.unary_union(perturbed_shape))
            differences.append(difference)
        return differences

    def _compute_gradient(self):
        grads = self._compute_adjoint_gradient_sweep(np.array([self.current_params[0]]))
        self.grad = grads[0] if grads.size == 1 else grads

    def _compute_adjoint_gradient_sweep(self, param_vals: np.ndarray, verbose: bool = False) -> np.ndarray:
        gradients = []
        for val in param_vals:
            self.current_params = np.array([val])
            self._update_qiskit_design()

            self._fwd_calculation()
            self.fwd_field_data = self._get_field_over_mesh(self.fwd_field_sParams)
            raw_E_at_JJ = self._get_field_at_junction(self.fwd_field_sParams)

            self.adjoint_source_location = [0, 0, 1e-6]

            base_shape = self.shapely_constructor.make_polygons(self.current_params)
            perturbed_params = self.current_params + self.param_perturbation
            perturbed_shape = self.shapely_constructor.make_polygons(perturbed_params)
            poly_base = shapely.unary_union(base_shape)
            poly_pert = shapely.unary_union(perturbed_shape)
            poly_grad = shapely.difference(poly_pert, poly_base)

            self._adjoint_calculation()
            adjoint_E = self.sim.eval_field_at_pts(self.adjoint_field_sParams, 'E', self.fwd_field_data["coords"])

            def calc_grad(point, poly_grad):
                point = np.real(point)
                dist = np.sqrt(shapely.distance(shapely.Point([point[0], point[1]]), poly_grad) ** 2 + point[2] ** 2)
                return np.exp(-0.5 * (dist / 0.002) ** 2)

            Ap_x = self.fwd_field_data['E'].copy()
            for j, mesh_coord in enumerate(self.fwd_field_data['coords']):
                Ap_x[j] *= calc_grad(mesh_coord, poly_grad)

            v_Ap_x = np.dot(adjoint_E.flatten(), Ap_x.flatten())
            gradients.append(v_Ap_x)
            if verbose:
                print(f"v_Ap_x = {np.real(v_Ap_x)} + {np.imag(v_Ap_x)}")

        return np.array(gradients)


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
