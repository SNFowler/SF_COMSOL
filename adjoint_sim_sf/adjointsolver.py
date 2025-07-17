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
    def get_design(self, design_parameters):
        pass

class SymmetricTransmonBuilder(DesignBuilder):
    """
    Produces a set of transmon designs that differ only by width.
    
    @TODO: Load constants from a config file rather than including the specifications directly in the design method.  
    """
    def __init__(self):
        pass

    def get_design(self, width):
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

        padCoordNums = [width, 0.02, 0.17926553, 0.25, 0.25]
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
    







        
