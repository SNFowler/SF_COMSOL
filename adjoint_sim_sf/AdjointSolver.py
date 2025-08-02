import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["PMIX_MCA_gds"]="hash"

# Import useful packages
import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict, open_docs

# To create plots after geting solution data.
import matplotlib.pyplot as plt
import numpy as np

# Packages for the simple design
from SQDMetal.Comps.Junctions import JunctionDolan
import shapely
from shapely.geometry import MultiPolygon, LineString, Point
from SQDMetal.Comps.Polygons import PolyShapely, PolyRectangle
from SQDMetal.Comps.Joints import Joint

from SQDMetal.COMSOL.Model import COMSOL_Model
from SQDMetal.COMSOL.SimCapacitance import COMSOL_Simulation_CapMats
from SQDMetal.COMSOL.SimRFsParameter import COMSOL_Simulation_RFsParameters

from SQDMetal.Utilities.ShapelyEx import ShapelyEx
from typing import Iterable

from adjoint_sim_sf import DesignBuilder, PolygonConstructor
    
class AdjointOptimiser:
    """
    Encapsulates the adjoint optimisation process.

    @TODO: The forward and backward solvers don't necessarily need to be explicitly computing the full array of E field values if they're already contained in the sim_sParams object. 
    @TODO: Decide on storing 2 entire COMSOL_Simulation_RFsParameters objects or just the fields.
    @TODO: Ensure old COMSOL simulations aren't kept open when running.
    """
    
    def __init__(self, initial_parameters: np.array, lr: float, qiskit_builder: DesignBuilder, shapely_constructor: PolygonConstructor):
        self.qiskit_builder = qiskit_builder
        self.shapely_constructor = shapely_constructor
        self.design = None
        self.current_params = initial_parameters      # : np.ndarray
        self.lr = lr # learning rate

        self.fwd_field_sParams = None
        self.adjoint_field_sParams = None
        
        self.current_Ap = None                      # Partial derivative of the design with respect to the parameters
        self.param_perturbation = 1e-5

        self.grad = None


        self.freq_value                 =  8.0333e9
        self.fwd_source_location        =  [-25e-3, 2e-3, 100e-6]
        self.adjoint_source_location    =  [0,0, 100e-6]
    
    def _run_comsol_sim(self, model_name: str, is_adjoint: bool = False):
        assert self.design

        self.design.rebuild()

        #Instantiate a COMSOL model
        cmsl = COMSOL_Model(model_name)

        #Create simulations to setup - in this case capacitance matrix and RF s-parameter
        sim_sParams = COMSOL_Simulation_RFsParameters(cmsl, adaptive='None')

        #(A) - Initialise model from Qiskit-Metal design object: design
        cmsl.initialize_model(self.design, [sim_sParams], bottom_grounded=True)

        cmsl.add_metallic(1, threshold=1e-12, fuse_threshold=1e-10)
        cmsl.add_ground_plane()
        cmsl.fuse_all_metals()

        sim_sParams.create_port_JosephsonJunction('junction', L_J=4.3e-9, C_J=10e-15, R_J=10e3)

        # sim_sParams.add_surface_current_source_region("dielectric", 0.5)
        # sim_sParams.add_surface_current_source_region("metals", 10e-6, 2)

        # edp_pts = ShapelyEx.get_points_uniform_in_polygon(poly1, 0.01,0.01)
        # for cur_pt in edp_pts:
        #     x, y = cur_pt[0]*0.001, cur_pt[1]*0.001 #Converting from mm to m
        #     sim_sParams.add_electric_point_dipole([x,y, 1e-6], 1, [0,0,1])
        if is_adjoint:
            sim_sParams.add_electric_point_dipole(self.adjoint_source_location, 1, [0,1,0])
        else:
            sim_sParams.add_electric_point_dipole(self.fwd_source_location, 1, [0,1,0])

        cmsl.fine_mesh_around_comp_boundaries(['pad1', 'pad2'], minElementSize=10e-6, maxElementSize=50e-6)

        cmsl.build_geom_mater_elec_mesh(skip_meshing=True, mesh_structure='Fine')

        sim_sParams.set_freq_values([self.freq_value])
        # cmsl.plot()
        sim_sParams.run()

        return sim_sParams

    def _fwd_calculation(self):
        """
        Returns raw mesh field data.
        """
        self.fwd_field_sParams = self._run_comsol_sim('FwdModel', is_adjoint=False)

    def _adjoint_calculation(self):
        """
        Returns the E field only, evaluated at the forward mesh coordinates.
        """
        self.adjoint_field_sParams = self._run_comsol_sim('AdjModel', is_adjoint=True)

    def _evaluate_ds(self):
        #lazy computation
        pass

    def _compute_gradient(self):
        # Evaluate adjoint field at the forward mesh
        assert self.fwd_field_data

        # Evaluate the fwd and ajoint field at the same points
        adjoint_field_data = {
            "coords"   :    self.fwd_field_data["coords"], # shallow copy
            "E"        :    self.adjoint_field_sParams.eval_field_at_pts('E', self.fwd_field_data['coords'])
        }

        pass             

    def _compute_Ap(self) -> list:
        
        base_shape = self.shapely_constructor.make_polygons(self.current_params)
        intersections = []

        for i in range(len(self.current_params)):
            perturbed_params = np.copy(self.current_params)
            perturbed_params[i] += self.param_perturbation
            perturbed_shape = self.shapely_constructor.make_polygons(perturbed_params)

            intersection = shapely.intersection(shapely.unary_union(base_shape), shapely.unary_union(perturbed_shape))
            intersections.append(intersection)

        return intersections

    def _compute_Ap_lazy(self):
        # delete this
    
    def _get_shape_velocity(self, poly_base: MultiPolygon, poly_pert: MultiPolygon, delta_p: float, n_pts=500):
        """
        Compute normal shape velocity at points along the boundary of the base shape.
        
        Returns a list of tuples (point, velocity_vector).
        """
        boundary_base = poly_base.boundary
        boundary_pert = poly_pert.boundary

        # Sample points along base boundary
        if isinstance(boundary_base, LineString):
            pts_base = [boundary_base.interpolate(t, normalized=True) for t in np.linspace(0, 1, n_pts)]
        else:
            pts_base = []
            for geom in boundary_base.geoms:
                pts_base += [geom.interpolate(t, normalized=True) for t in np.linspace(0, 1, n_pts // len(boundary_base.geoms))]

        # For each base boundary point, find nearest point on perturbed boundary
        velocity_vectors = []
        for pt in pts_base:
            closest = boundary_pert.interpolate(boundary_pert.project(pt))
            velocity = np.array(closest.coords[0]) - np.array(pt.coords[0])
            velocity_vectors.append((pt, velocity / delta_p))

        return velocity_vectors


    def optimisation(self):
        # initialise optimizer
        # intialize training data log
        # training loop:
            # if training condition satisfied:
                #break
            #compute gradient:
                # run forward calc
                # compute adjoint source
                # run adjoint calc
                # compute A_p
                # compute Gradient Function
            # update parameters
            # update training data log

        # Return final design, training data

        pass








        
