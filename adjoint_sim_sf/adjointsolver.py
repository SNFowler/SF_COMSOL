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
from SQDMetal.Comps.Polygons import PolyShapely, PolyRectangle
from SQDMetal.Comps.Joints import Joint

from SQDMetal.COMSOL.Model import COMSOL_Model
from SQDMetal.COMSOL.SimCapacitance import COMSOL_Simulation_CapMats
from SQDMetal.COMSOL.SimRFsParameter import COMSOL_Simulation_RFsParameters

from SQDMetal.Utilities.ShapelyEx import ShapelyEx
from typing import Iterable


class DesignBuilder():
    """
    Interface for producing Qiskit design from an np array of parameters. 
    """
    def get_design(self, design_parameters : np.ndarray):
        pass

class SymmetricTransmonBuilder(DesignBuilder):
    """
    Produces a set of transmon designs that differ only by width.
    
    @TODO: Load constants from a config file rather than including the specifications directly in the design method.  
    """
    def __init__(self):
        pass

    def get_design(self, width: np.ndarray):
        """
        Outputs the Qiskit design given a single width parameter.

        This method is extremely impure.
        """
        
        # Procedure from SweepPoly2.ipynb
        # Set up chip design as planar, multiplanar also available
        design = designs.DesignPlanar({}, overwrite_enabled=True)

        # Set up chip dimensions 
        design.chips.main.size.size_x = '800um'
        design.chips.main.size.size_y = '800um'
        design.chips.main.size.size_z = '500um'
        design.chips.main.size.center_x = '0mm'
        design.chips.main.size.center_y = '0mm'

        JunctionDolan(design, 'junction', options=Dict(pos_x=0, pos_y='-12um', end_x=0, end_y='12um',
                                                                layer=2,
                                                                finger_width='0.4um', t_pad_size='0.385um',
                                                                squid_width='5.4um', prong_width='0.9um'));

        padCoordNums = [width[0], 0.02, 0.17926553, 0.25, 0.25]
        padCoords = [[-0.05, 0.012], [0.05, 0.012], [padCoordNums[0], padCoordNums[1]], [padCoordNums[2], padCoordNums[3]], [0, padCoordNums[4]], [-padCoordNums[2], padCoordNums[3]], [-padCoordNums[0], padCoordNums[1]]]
        padCoords2 = [[x[0],-x[1]] for x in padCoords][::-1]

        poly1 = shapely.Polygon(padCoords).buffer(-0.04, join_style=1, quad_segs=4).buffer(0.04, join_style=1, quad_segs=4)
        poly2 = shapely.Polygon(padCoords2).buffer(-0.04, join_style=1, quad_segs=4).buffer(0.04, join_style=1, quad_segs=4)

        PolyShapely(design, 'pad1', options=dict(strShapely=poly1.__str__()))
        PolyShapely(design, 'pad2', options=dict(strShapely=poly2.__str__()))

        design.rebuild()

        PolyRectangle(design, 'rectGnd', options=dict(pos_x='-300um',pos_y='-300um', end_x='300um',end_y='300um', is_ground_cutout=True))

        Joint(design, 'j1', options=dict(pos_x='0um', pos_y='-300um'))
        Joint(design, 'j2', options=dict(pos_x='0um', pos_y='300um'));

        #ebuild the GUI
        return design
    
    def view_design(self, design):
        gui = MetalGUI(design)
        gui.rebuild()
    
    
class AdjointOptimiser:
    """
    Encapsulates the adjoint optimisation process.

    @TODO: The forward and backward solvers don't necessarily need to be explicitly computing the full array of E field values if they're already contained in the sim_sParams object. 
    @TODO: Decide on storing 2 entire COMSOL_Simulation_RFsParameters objects or just the fields.
    @TODO: Ensure old COMSOL simulations aren't kept open when running.
    """
    
    def __init__(self, initial_parameters: np.array, lr: float, builder: DesignBuilder):
        self.builder = builder
        self.design = None
        self.current_params = self.initial_parameters      # : np.ndarray
        self.lr = lr # learning rate

        self.fwd_field_sParams = None
        self.adjoint_field_sParams = None
        self.dA_dparams

        self.grad = None

        # Note that currently the fwd_field_data contains coordinates, E, B, D and H field while the adjoint_field_data contains only coordinates and E.
        self.fwd_field_data = None
        self.adjoint_field_data = None

        self.fwd_source_location =      [-25e-3, 2e-3, 100e-6]
        self.adjoint_source_location =  [0,0, 100e-6]


    def _fwd_calculation(self):
        assert self.design

        self.design.rebuild()

        #Instantiate a COMSOL model
        cmsl = COMSOL_Model('FwdModel')

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

        sim_sParams.add_electric_point_dipole(self.fwd_source_location, 1, [0,1,0])
        # sim_sParams.add_electric_point_dipole([0,0,0], 0, [0,1,0])

        cmsl.fine_mesh_around_comp_boundaries(['pad1', 'pad2'], minElementSize=10e-6, maxElementSize=50e-6)

        cmsl.build_geom_mater_elec_mesh(skip_meshing=True, mesh_structure='Fine')

        sim_sParams.set_freq_values([8.0333e9])
        # cmsl.plot()
        sim_sParams.run()
        
        return sim_sParams.eval_fields_over_mesh


    def _adjoint_calculation(self):
        assert self.design
        assert self.fwd_field_data

        self.design.rebuild()

        #Instantiate a COMSOL model
        cmsl = COMSOL_Model('AdjModel')

        #Create simulations to setup - in this case capacitance matrix and RF s-parameter
        adj_sParams = COMSOL_Simulation_RFsParameters(cmsl, adaptive='None')

        #(A) - Initialise model from Qiskit-Metal design object: design
        cmsl.initialize_model(self.design, [adj_sParams], bottom_grounded=True)

        cmsl.add_metallic(1, threshold=1e-12, fuse_threshold=1e-10)
        cmsl.add_ground_plane()
        cmsl.fuse_all_metals()

        adj_sParams.create_port_JosephsonJunction('junction', L_J=4.3e-9, C_J=10e-15, R_J=10e3)

        # sim_sParams.add_surface_current_source_region("dielectric", 0.5)
        # sim_sParams.add_surface_current_source_region("metals", 10e-6, 2)

        # edp_pts = ShapelyEx.get_points_uniform_in_polygon(poly1, 0.01,0.01)
        # for cur_pt in edp_pts:
        #     x, y = cur_pt[0]*0.001, cur_pt[1]*0.001 #Converting from mm to m
        #     adj_sParams.add_electric_point_dipole([x,y, 1e-6], 1, [0,0,1])

        adj_sParams.add_electric_point_dipole(self.adjoint_source_location, 1, [0,1,0])
        # adj_sParams.add_electric_point_dipole([0,0,0], 0, [0,1,0])

        cmsl.fine_mesh_around_comp_boundaries(['pad1', 'pad2'], minElementSize=10e-6, maxElementSize=50e-6)

        cmsl.build_geom_mater_elec_mesh(skip_meshing=True, mesh_structure='Fine')

        adj_sParams.set_freq_values([8.0333e9])
        # cmsl.plot()
        adj_sParams.run()

        # Evaluate adjoint field at the forward mesh
        adjoint_field_data = {
            "coords"   :    self.fwd_field_data["coords"], # shallow copy
            "E"        :    adj_sParams.eval_field_at_pts('E', self.fwd_field_data['coords'])
        }

        adjoint_field_values = adj_sParams.eval_field_at_pts('E', self.fwd_field_data['coords'])
        


    def _evaluate_ds(self):
        #lazy computation
        pass

    def _compute_gradient(self):
        pass             #A_p

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








        
