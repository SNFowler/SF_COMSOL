from abc import ABC, abstractmethod
import os
from math import hypot
from typing import List, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import shapely
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points
from shapely.plotting import plot_polygon

import qiskit_metal as metal
from qiskit_metal import Dict, MetalGUI, designs, draw

from SQDMetal.Comps.Junctions import JunctionDolan
from SQDMetal.Comps.Polygons import PolyRectangle, PolyShapely
from SQDMetal.Comps.Joints import Joint

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

############################################################
###                 INTERFACES                           ###
############################################################

class ParametricDesign(ABC):
    @abstractmethod
    def build_qk_design(self, parameters):
        pass

    @abstractmethod
    def geometry(self, parameters):
        pass

    @abstractmethod
    def compute_Ap(self, parameters, perturbation):
        pass

    @abstractmethod
    def compute_boundary_velocity(self, parameters, perturbation):
        """Return boundary velocity info for a small parameter perturbation."""
        pass

    @abstractmethod
    def sample_interior_points(self, params: np.ndarray, n: int, seed=None):
        pass


class PolygonConstructor(ABC):
    @abstractmethod
    def make_polygons(self, design_parameters):
        pass

    def compute_boundary_velocity(self, design_parameters, perturbation):
        """Optional default; concrete classes may override."""
        raise NotImplementedError


class DesignBuilder(ABC):
    @abstractmethod
    def get_design(self, shapely_design):
        pass


############################################################
###                 IMPLEMENTATIONS                      ###
############################################################

class SymmetricTransmonDesign(ParametricDesign):
    def __init__(self):
        self._polygon_constructor = SymmetricTransmonPolygonConstructor()
        self._design_builder = SymmetricTransmonBuilder()

    def build_qk_design(self, parameters):
        geom = self._polygon_constructor.make_polygons(parameters)
        return self._design_builder.get_design(geom)
    
    def show_design(self, parameters):
        design = self.build_qk_design(parameters)
        return self._design_builder.view_design(design)

    def show_polygons(self, parameters) -> Figure:
        """
        Returns a Figure object, does not display the figure automatically when outside of a notebook.
        """
        
        return self._polygon_constructor.show_polygons(parameters)


    def geometry(self, parameters):
        return self._polygon_constructor.make_polygons(parameters)

    def compute_Ap(self, parameters, perturbation):
        """
        Compute difference between perturbed and base geometries.
        """
        base_geom = shapely.unary_union(self.geometry(parameters))
        pert_geom = shapely.unary_union(self.geometry(parameters + perturbation))
        return shapely.difference(pert_geom, base_geom)
    
    def compute_boundary_velocity(self, parameters: np.ndarray, perturbation: np.ndarray):
        return self._polygon_constructor.compute_boundary_velocity(parameters, perturbation)
    
    def sample_interior_points(self, params: np.ndarray, n: int, seed=None):
        """Sample n points from interior of design geometry."""
        return self._polygon_constructor.sample_interior_points(params, n, seed)

class SymmetricTransmonPolygonConstructor(PolygonConstructor):
    def __init__(self, join_style=1, quad_segs=4):
        self.join_style = join_style
        self.quad_segs = quad_segs

    def make_polygons(self, params):
        """
        Returns a multipolygon object.
        Units are in mm, converted to m downstream of this class.
        """
        width = params[0]
        height = params[1] if len(params) > 1 else 0.25


        padCoordNums = [width, 0.02, 0.17926553, height, height] #mm

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

        return shapely.MultiPolygon([poly1, poly2])
    
    def show_polygons(self, parameters) -> Figure:
        polys = self.make_polygons(parameters)
        fig, ax = plt.subplots()
        for poly in polys:
            if poly.geom_type == "Polygon":
                plot_polygon(poly, ax=ax)
            else:
                for p in poly.geoms:
                    plot_polygon(p, ax=ax)
        ax.set_aspect("equal")
        return fig
    
    def compute_boundary_velocity(self, params: np.ndarray, perturbation_vec: np.ndarray):
        reference_multipoly = self.make_polygons(params)
        perturbed_multipoly = self.make_polygons(params + perturbation_vec)

        epsilon = np.linalg.norm(perturbation_vec)

        if epsilon == 0:
            raise ValueError("magnitude of perturbation must be nonzero")

        polys = [reference_multipoly] if isinstance(reference_multipoly, Polygon) else list(reference_multipoly.geoms)
        perturbed_boundary = perturbed_multipoly.boundary

        velocities = []                 # [poly][ring][i]
        reference_coords = []           # [poly][ring][i] -> (x,y)
        nearest_perturbed_coords = []   # [poly][ring][i] -> (x,y)

        for poly in polys:
            rings = [poly.exterior] + list(poly.interiors)

            poly_vels = []
            poly_refs = []
            poly_corrs = []

            for ring in rings:
                coords = list(ring.coords)
                if len(coords) > 1 and coords[0] == coords[-1]:
                    coords = coords[:-1]

                ring_vels = []
                ring_refs = []
                ring_corrs = []

                for x_i, y_i in coords:
                    p = Point(x_i, y_i)
                    q, _ = nearest_points(perturbed_boundary, p)

                    d = hypot(q.x - x_i, q.y - y_i) 
                    sign = -1 if reference_multipoly.covers(q) else 1
                    ring_vels.append(sign * d / epsilon)

                    ring_refs.append((x_i, y_i))
                    ring_corrs.append((q.x, q.y))

                poly_vels.append(ring_vels)
                poly_refs.append(ring_refs)
                poly_corrs.append(ring_corrs)

            velocities.append(poly_vels)
            reference_coords.append(poly_refs)
            nearest_perturbed_coords.append(poly_corrs)

        return velocities, reference_coords, nearest_perturbed_coords
    
    def sample_interior_points(self, params: np.ndarray, n: int, seed=None) -> np.ndarray:
        """
        Sample n points uniformly from the interior of the design.
        Returns array of shape (n, 2).
        """
        multipoly = self.make_polygons(params)
        
        # Get bounding box
        minx, miny, maxx, maxy = multipoly.bounds
        
        # Sample with rejection
        points = []
        rng = np.random.default_rng(seed)
        
        while len(points) < n:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            if multipoly.contains(Point(x, y)):
                points.append([x * 1e-3, y * 1e-3])  # Convert mm to m
        
        return np.array(points)



class SymmetricTransmonBuilder(DesignBuilder):
    def get_design(self, multipoly):
        import shapely


        assert shapely.get_num_geometries(multipoly) == 2
        poly1, poly2 = multipoly.geoms

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
        try:
            from IPython import get_ipython
            if get_ipython() and 'IPKernelApp' in get_ipython().config:
                gui = MetalGUI(design)
                gui.rebuild()
            else:
                raise RuntimeError("view_design() only works inside a Jupyter notebook environment.")
        except Exception:
            raise RuntimeError("view_design() only works inside a Jupyter notebook environment.")

 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    width = [0.19971691]
    designer = SymmetricTransmonDesign()
    fig = designer.show_polygons(width)
    plt.show(block=True) 
