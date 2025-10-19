"""
AdjointSolver.py
"""

import os
import math
import time
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
from typing import Iterable, List
from dataclasses import dataclass, fields, asdict

from .ParametricDesign import ParametricDesign
from .Simulation import SimulationRunner 
from .Sources import Source 
"""
@TODO: Naming confusion. "design" in this code refers to both the qiskit design and to an instance of ParametricDesign class.
@TODO: Account for changes in loss directly due to design parameters.
"""    

class AdjointEvaluator:
    def __init__(self, parametric_designer: ParametricDesign, config: dict = None):
        if COMSOL_Model._engine is None:
            raise RuntimeError(
                "COMSOL engine not initialized. "
                "Call COMSOL_Model.init_engine() before running simulations."
            )
        
        # debugging states
        self.verbose = True
        self.print_timing = True
        
        self.random_seed = None

        # self.freq_value = 8.0333e9
        self.freq_value = 8e9
        self.param_perturbation = np.array([1e-5])          # in mm
        self.fwd_source_strength = 1e-2
        self.silicon_base_area = 3.6E-7                     # in m, 600um*600um

        self.num_adj_sample_points = 20
        self.source_vertical_displacement = 100e-6 # sources can't be at the surface exactly.
        
        self.fwd_source_locations = np.array([[300e-6, 300e-6, self.source_vertical_displacement], [-200e-6, 400e-6, self.source_vertical_displacement]])  # in meters
        self.current_adjoint_sources = [
            Source(np.array([0, 0, self.source_vertical_displacement]), np.array([0, 1, 0]))
        ]

        self.canonical_jj_adjoint_source = [
            Source(np.array([0, 0, self.source_vertical_displacement]), np.array([0, 1, 0]))
        ]
        
        
              # in meters
        self.adjoint_rotation = math.pi/2 
        self.param_to_sim_scale = 1e-3

        self.parametric_designer = parametric_designer
        if config is not None:
            self.update_params(config)

        self.sim_runner = SimulationRunner(self.freq_value)

    def _fwd_calculation(self, design):
        # construct source objects
        sources = [Source(loc, np.array([0, 1, 0]), self.fwd_source_strength) for loc in self.fwd_source_locations]
        

        starttime = time.time()
        res = self.sim_runner.run_forward(design, sources)
        endtime = time.time()

        if self.print_timing: print(f"forward sim time : {endtime - starttime:.6f}")

        return res

    def _adjoint_calculation(self, design, fwd_sparams):
        str_values = self._adjoint_strength(fwd_sparams, self.current_adjoint_sources)
        
        # Create new Source objects with computed strengths
        sources = [
            Source(src.location, src.direction, strength)
            for src, strength in zip(self.current_adjoint_sources, str_values)
        ]

        starttime = time.time()
        res = self.sim_runner.run_adjoint(design, sources)
        endtime = time.time()
        if self.print_timing:  print(f"adjoint sim time : {endtime - starttime:.6f}")
        
        return self.sim_runner.run_adjoint(design, sources)
    
    def _adjoint_strength(self, fwd_sparams, adjoint_sources):
        """
        Evaluates the adjoint strength scalars for each source.
        
        Args:
            fwd_sparams: Forward simulation results
            adjoint_sources: List of Source objects (without strength set)
        """
        locations = np.array([src.location for src in adjoint_sources])
        directions = np.array([src.direction for src in adjoint_sources])
        
        raw_E = self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', locations)
        mu_0 = 1.25663706127e-6
        
        # Dot product: E · direction for each source
        E_dot_direction = np.sum(raw_E * directions, axis=-1)
        
        adj_strength = (
            2 * np.real(E_dot_direction) 
            / (2 * np.pi * self.freq_value * mu_0)
        )
        
        return adj_strength
  
    def evaluate(self, params, perturbation_magnitude, verbose=True, 
             objective_type="jj", fwd_sparams=None, adj_sparams=None):
        """Evaluate the gradient and loss at given params."""
    
        # Get objective-specific info
        sources, norm_area = self._get_objective_info(params, objective_type)
        self.current_adjoint_sources = sources
        
        # Everything else is the same for all objectives
        qk_design = self.parametric_designer.build_qk_design(params)
        
        if fwd_sparams is None:
            
            fwd_sparams = self._fwd_calculation(qk_design)
            
            
        if adj_sparams is None:
            adj_sparams = self._adjoint_calculation(qk_design, fwd_sparams)
        
        # Compute gradient and loss (shared for all)
        starttime = time.time()
        grad_vec, loss = self._compute_loss_and_gradient(
            params, perturbation_magnitude, sources, 
            fwd_sparams, adj_sparams, norm_area
        )
        
        endtime = time.time()

        if self.print_timing:  print(f"inner product time : {endtime - starttime:.6f}")     
        if verbose:
            print(f"{objective_type}: {len(sources)} sources, loss={loss:.3e}")
        
        return grad_vec, loss
    
    def evaluate_multi_objective(self, params, perturbation_magnitude, 
                                w_jj=0.5, 
                                w_sa=0.5, 
                                w_ma=0.0,
                                verbose = True):
        # Run forward sim once
        qk_design = self.parametric_designer.build_qk_design(params)
        fwd_sparams = self._fwd_calculation(qk_design)


        grad_jj = grad_sa = grad_ma = np.zeros_like(params)
        loss_jj = loss_sa = loss_ma = 0.0

        
        # JJ objective
        if w_jj > 0:
            grad_jj, loss_jj = self.evaluate(params, perturbation_magnitude, 
                                                fwd_sparams=fwd_sparams)
       
        # SA objective 
        if w_sa > 0:
            grad_sa, loss_sa = self.evaluate(params, perturbation_magnitude,
                                                    fwd_sparams=fwd_sparams, objective_type="sa")
        
        # MA objective, if used
        if w_ma > 0:
            grad_ma, loss_ma = self.evaluate(params, perturbation_magnitude,
                                                    fwd_sparams=fwd_sparams, objective_type="ma")
        
        # Combine
        total_grad = w_jj * grad_jj + w_sa * grad_sa + w_ma * grad_ma
        total_loss = w_jj * loss_jj + w_sa * loss_sa + w_ma * loss_ma

        #@TODO: Add scaling normalisation to epr objective.
       
        current_results = {
            "params": np.asarray(params, float),
            "loss": float(total_loss),
            "grad": np.asarray(total_grad, float),
            "loss_jj": loss_jj,
            "loss_sa": loss_sa,
            "loss_ma": loss_ma,
            "grad_jj": grad_jj,
            "grad_sa": grad_sa,
            "grad_ma": grad_ma
        }

        
        return current_results
    
    def _compute_loss_and_gradient(self, params, perturbation_magnitude, 
                               adjoint_sources, fwd_sparams, adj_sparams,
                               integral_area=None):
        """
        Core gradient/loss computation given sources.

        Currently ignores pure design parameter dependence of loss, i.e. penalising large values of theta directly.

        \frac{dL}{d\theta} &= 
            \frac{\partial \mathcal{L}}{\partial \theta} + # assumed zero for now
          + \int_{\partial \Omega} \textit{some constants} \quad u(s) \mathbf{v}^\dagger \mathbf{E}\quad ds +
          + \int_{\partial \Omega} u(s) \mathbf{E_z}^\dagger \mathbf{E_z}\quad ds
        
        Args:
            normalization_area: Optional area for loss scaling (e.g., MA or SA surface area)
        """
        grad_vec = np.zeros_like(params)

        # --- Gradient Calculation ---
        # TODO: Can be made vastly more efficient by computing the adjoint and E field inner product as a scalar field,
        #       then integrating over the boundary with the velocity field. 
        #       Likewise, computing the |E_z|^2 field as a scalar field also.


        #  Part 1 - Adjoint method
        # \int_{\partial \Omega} \textit{some constants} \quad u(s) \mathbf{v}^\dagger \mathbf{E}\quad ds 
        for basis_index in range(params.shape[0]):
            perturbation_vec = np.zeros_like(params)
            perturbation_vec[basis_index] = perturbation_magnitude
            
            #TODO: Compute multiple inner products at once for efficiency.
            inner_product = self.compute_boundary_inner_product(
                params, perturbation_vec, fwd_sparams, adj_sparams
            )
            
            grad_complex = -inner_product
            grad = 2.0 * np.real(grad_complex)
            grad_vec[basis_index] = grad

        # Part 2 - Reynolds Transport Component
        # \int_{\partial \Omega} u(s) \mathbf{E_z}^\dagger \mathbf{E_z}\quad ds
        for basis_index in range(params.shape[0]):
            perturbation_vec = np.zeros_like(params)
            perturbation_vec[basis_index] = perturbation_magnitude

            boundary_velocity_field, reference_coord, _, ds = self.parametric_designer.compute_boundary_velocity(params, perturbation_vec)

        

        # --- Loss Calculation ---
        
        loss = self.compute_loss(params, fwd_sparams)
        
        # Multiply by area
        if integral_area is not None:
            loss = loss * integral_area
        
        return grad_vec, loss
    
    def compute_loss(self, params, fwd_sparams=None):
        """Compute only the loss at given params."""
        qk_design = self.parametric_designer.build_qk_design(params)
        
        if fwd_sparams is None:
            fwd_sparams = self._fwd_calculation(qk_design)
        
        # Compute loss
        locations = np.array([src.location for src in self.current_adjoint_sources])
        directions = np.array([src.direction for src in self.current_adjoint_sources])
        E_at_targets = self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', locations)
        E_projected = np.sum(E_at_targets * directions, axis=-1)
        
        loss = -np.sum(np.abs(E_projected)**2)
        
        return loss
    
   


    def _calc_adjoint_forward_product(self, 
                                      boundary_velocity_field,
                                      reference_coord,
                                      fwd_sparams: COMSOL_Simulation_RFsParameters, 
                                      adj_sparams: COMSOL_Simulation_RFsParameters,
                                      adjoint_rotation: float = 0):
        

        dr = 5e-6 # m
         #TODO: extract this from the boundary field intelligently, or pass it from/to parametric designer -> polygon constructor

        running_sum = 0

        #TODO: Enforce non-2D
        r3 = 1e-6

        #TODO: Vectorise this entire process and inner product for efficiency.

        flat_boundary_velocity_field = [
                                        v for poly_vels in boundary_velocity_field
                                        for ring_vels in poly_vels
                                        for v in ring_vels
                                        ]
        flat_reference_coord =          [
                                    xy for poly_refs in reference_coord
                                    for ring_refs in poly_refs
                                    for xy in ring_refs
                                        ]
        

        phase = np.exp(1j * float(adjoint_rotation))

        if self.print_timing: start_inner_time = time.time()
        for v, (r1, r2) in zip(flat_boundary_velocity_field, flat_reference_coord):
            # complete the forward and adjoint vectors. Note that the adjoint vector is actually the complex conjugate (but not the transpose) of the expected adjoint vector.

            local_fwd_vec =         self.sim_runner.eval_field_at_pts(fwd_sparams, 'E', [[r1, r2, r3]])
            local_adj_vec_conj =    self.sim_runner.eval_field_at_pts(adj_sparams, 'E', [[r1, r2, r3]])  # extract E value of fwd_sparams at r1, r2
            
            local_adj_vec_conj = local_adj_vec_conj * phase

            # Taking the inner product. Note the vectors are intially in row form. 
            running_sum += v*(local_adj_vec_conj@local_fwd_vec.T)*dr

        if self.print_timing: end_inner_time = time.time(); print(f"inner product loop time : {end_inner_time - start_inner_time:.6f}")
        return running_sum 

    def compute_boundary_inner_product(self, params, perturbation, fwd_sparams, adj_sparams):
        """
        Compute the boundary velocity inner product between forward and adjoint fields.
        
        Args:
            params: Current parameter values
            perturbation: Perturbation for boundary velocity calculation
            fwd_sparams: Forward simulation results
            adj_sparams: Adjoint simulation results
            
        Returns:
            Complex-valued inner product
        """
        boundary_velocity_field, reference_coord, _, ds = \
            self.parametric_designer.compute_boundary_velocity(params, perturbation)
        
        return self._calc_adjoint_forward_product(
            boundary_velocity_field, reference_coord, 
            fwd_sparams, adj_sparams, 
            adjoint_rotation=self.adjoint_rotation
        )
    
    def _get_objective_info(self, params, objective_type):
        """Returns (sources, normalization_area) for the objective"""
        
        match objective_type.lower():
            case "jj":
                sources = self.get_JJ_adjoint_sources()
                area = None
            
            case "sa":
                sources = self.get_epr_adjoint_sources(params, self.num_adj_sample_points, 
                                                    interface="SA", seed=self.random_seed)
                area = self._get_silicon_area(params)
            
            case "ma":
                sources = self.get_epr_adjoint_sources(params, self.num_adj_sample_points,
                                                    interface="MA", seed=self.random_seed)
                area = self._get_metal_area(params)
            
            case "manual":
                sources = self.current_adjoint_sources  # Already set
                area = None
            
            case _:
                raise ValueError(f"Unknown objective: {objective_type}")
    
        return sources, area

    def save(self):
        self.sim_runner.save()

    def _calc_Ap(self, current_param, fwd_field_data, perturbation):
        """
        WIP: In the future will facilitate non-boundary velocity methods
        """
        raise NotImplementedError
    
    


    def visualise(self, sims_params):
        field_data = self.sim_runner.eval_fields_over_mesh(sims_params)
        import matplotlib.pyplot as plt

        x,y,z,Ez = field_data['coords'][:,0], field_data['coords'][:,1], field_data['coords'][:,2], field_data['E'][:,2]

        plane_inds = (np.abs(z)<1e-6)
        plt.scatter(x[plane_inds], y[plane_inds], c=np.clip(np.abs(Ez[plane_inds]), 0,1e14))

    def get_JJ_adjoint_sources(self):
        return self.canonical_jj_adjoint_source
    
    def get_epr_adjoint_sources(self, params, n, interface = "SA", seed=None):
        """
        Returns a random sample of points above the design region without strengths
        """
        iface = interface.upper()
        assert iface in {"MA", "SA"}, f"Invalid interface '{iface}'" 
        
        vertical_displacement = 1e-6
        orientation = np.array([0,0,1])
        if iface == "MA":
            points_2d = self.parametric_designer.sample_MA_points(params, n, seed=seed)
        if iface == "SA":
            points_2d = self.parametric_designer.sample_SA_points(params, n, seed=seed)
            
        sources = [Source(np.array([pt[0], pt[1], vertical_displacement]), orientation) for pt in points_2d]

        return sources
    
    # -----
    # Utilities: 
    #   - 1. geometry
    #   - 2. updating/outputting params
    # -----

    # - 1. geometry
    
    def _get_metal_area(self, params: np.ndarray) -> float:
        """
        Area of the pad geometry.Does not include the metal square surrounding the design region.
        """
        area_m2 = self.parametric_designer.get_interior_area(params)

        return area_m2
    
    def _get_silicon_area(self, params: np.ndarray) -> float:
        # the original silicon area minus the pads area.
        return self.silicon_base_area - self._get_metal_area(params)
    
    def plot_sources(self, sources: List[Source], arrow_scale = 0.25):

        # unit conversion
        conversion = 1e4

        xs = np.array([s.location[0] for s in sources])
        ys = np.array([s.location[1] for s in sources])
        zs = np.array([s.location[2] for s in sources])
        us = np.array([s.direction[0] for s in sources])
        vs = np.array([s.direction[1]  for s in sources])
        ws = np.array([s.direction[2]  for s in sources])

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        ax1.scatter(xs, ys, s=20)
        ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_title("Source XY positions")
        ax1.axis("equal")
        

        #rescale for the quiver plot to work properly
        xs = np.array([s.location[0]*conversion for s in sources])
        ys = np.array([s.location[1]*conversion for s in sources])
        zs = np.array([s.location[2]*conversion for s in sources])
        
        ax2.quiver(xs, ys, zs, us, vs, ws, length=arrow_scale, arrow_length_ratio=0.2, linewidth=1)
       
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.set_title("Source directions")

        ax2.set_xlim([-4, 4])
        ax2.set_ylim([-4, 4])
        ax2.set_zlim([0, 8])

        ax2.set_box_aspect([1, 1, 1])


        return fig, (ax1, ax2)


    # - 2. updating/outputting params

    def update_params(self, config_dict):
        for k, v in config_dict.items():
            if not hasattr(self, k):
                raise AttributeError(f"AdjointEvaluator has no attribute {k}")
            if isinstance(v, list):
                v = np.array(v)
            setattr(self, k, v)


    def to_config_dict(self):
        cfg = {}
        for key, val in vars(self).items():
            if key.startswith("_") or callable(val):
                continue
            if key in {"parametric_designer", "sim_runner"}:
                continue

            if isinstance(val, np.ndarray):
                cfg[key] = val.tolist()
            elif isinstance(val, Source):
                cfg[key] = asdict(val)
            elif isinstance(val, list) and val and isinstance(val[0], Source):
                cfg[key] = [asdict(v) for v in val]
            else:
                cfg[key] = val
        return cfg


