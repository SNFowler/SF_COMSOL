from abc import ABC, abstractmethod
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict
import numpy as np
import shapely
from shapely.geometry import Polygon
from SQDMetal.Comps.Junctions import JunctionDolan
from SQDMetal.Comps.Polygons import PolyShapely, PolyRectangle
from SQDMetal.Comps.Joints import Joint

############################################################
###                 INTERFACES                           ###
############################################################

class ParametricDesign(ABC):
    @abstractmethod
    def build_design(self, parameters):
        """Convert parameters directly to a Qiskit design."""
        pass

    @abstractmethod
    def geometry(self, parameters):
        """Return the base geometry for given parameters."""
        pass

    @abstractmethod
    def compute_Ap(self, parameters, perturbation):
        """
        Compute the shape derivative (A_p) given parameters and a perturbation size.
        The return type is deliberately untyped so it can be shapely or other formats.
        """
        pass


class PolygonConstructor(ABC):
    @abstractmethod
    def make_polygons(self, design_parameters):
        """Generate polygons based on a tuple of design parameters."""
        pass


class DesignBuilder(ABC):
    @abstractmethod
    def get_design(self, shapely_design):
        pass


############################################################
###                 IMPLEMENTATIONS                      ###
############################################################

class SymmetricTransmonDesign(ParametricDesign):
    def __init__(self):
        self.polygon_constructor = SymmetricTransmonPolygonConstructor()
        self.design_builder = SymmetricTransmonBuilder()

    def build_design(self, parameters):
        polygons = self.polygon_constructor.make_polygons(parameters)
        return self.design_builder.get_design(polygons)

    def geometry(self, parameters):
        return self.polygon_constructor.make_polygons(parameters)

    def compute_Ap(self, parameters, perturbation):
        """
        Compute difference between perturbed and base geometries.
        """
        base_geom = shapely.unary_union(self.geometry(parameters))
        pert_geom = shapely.unary_union(self.geometry(parameters + perturbation))
        return shapely.difference(pert_geom, base_geom)


class SymmetricTransmonPolygonConstructor(PolygonConstructor):
    def __init__(self, join_style=1, quad_segs=4):
        self.join_style = join_style
        self.quad_segs = quad_segs

    def make_polygons(self, params):
        width = params[0]
        padCoordNums = [width, 0.02, 0.17926553, 0.25, 0.25]

        padCoords = [
            [-0.05, 0.012], 
            [0.05, 0.012],
            [padCoordNums[0], padCoordNums[1]],
            [padCoordNums[2], padCoordNums[3]],
            [0, padCoordNums[4]],
            [-padCoordNums[2], padCoordNums[3]],
            [-padCoordNums[0], padCoordNums[1]]
        ]
        padCoords2 = [[x[0], -x[1]] for x in padCoords][::-1]

        poly1 = Polygon(padCoords).buffer(-0.04, join_style=self.join_style, quad_segs=self.quad_segs)
        poly1 = poly1.buffer(0.04, join_style=self.join_style, quad_segs=self.quad_segs)

        poly2 = Polygon(padCoords2).buffer(-0.04, join_style=self.join_style, quad_segs=self.quad_segs)
        poly2 = poly2.buffer(0.04, join_style=self.join_style, quad_segs=self.quad_segs)

        return poly1, poly2


class SymmetricTransmonBuilder(DesignBuilder):
    def get_design(self, shapely_components):
        assert len(shapely_components) == 2

        design = designs.DesignPlanar({}, overwrite_enabled=True)
        design.chips.main.size.size_x = '800um'
        design.chips.main.size.size_y = '800um'
        design.chips.main.size.size_z = '500um'
        design.chips.main.size.center_x = '0mm'
        design.chips.main.size.center_y = '0mm'

        JunctionDolan(design, 'junction', options=Dict(
            pos_x=0, pos_y='-12um', end_x=0, end_y='12um',
            layer=2, finger_width='0.4um', t_pad_size='0.385um',
            squid_width='5.4um', prong_width='0.9um'))

        poly1, poly2 = shapely_components
        PolyShapely(design, 'pad1', options=dict(strShapely=poly1.__str__()))
        PolyShapely(design, 'pad2', options=dict(strShapely=poly2.__str__()))

        design.rebuild()
        PolyRectangle(design, 'rectGnd', options=dict(
            pos_x='-300um', pos_y='-300um',
            end_x='300um', end_y='300um',
            is_ground_cutout=True))

        Joint(design, 'j1', options=dict(pos_x='0um', pos_y='-300um'))
        Joint(design, 'j2', options=dict(pos_x='0um', pos_y='300um'))

        return design

    def view_design(self, design):
        gui = MetalGUI(design)
        gui.rebuild()
