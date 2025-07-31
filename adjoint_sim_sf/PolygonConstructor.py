from abc import ABC, abstractmethod
from shapely.geometry import Polygon
from typing import Tuple
import numpy as np

class PolygonConstructor(ABC):
    @abstractmethod
    def make_polygons(self, design_parameters: np.ndarray):
        """
        Generate Shapely polygons based on a tuple of design parameters.
        """
        pass

class SymmetricTransmonPolygonConstructor(PolygonConstructor):
    """
    Concrete implementation of PolygonConstructor that builds two symmetric
    transmon pads using a single width parameter.
    """

    def __init__(self, join_style: int = 1, quad_segs: int = 4):
        self.join_style = join_style
        self.quad_segs = quad_segs

    def make_polygons(self, params: np.ndarray) -> Tuple[Polygon, Polygon]:
        width = params[0]
        padCoordNums = [width, 0.02, 0.17926553, 0.25, 0.25]

        padCoords = [   [-0.05, 0.012], 
                        [0.05, 0.012],
                        [padCoordNums[0], padCoordNums[1]],
                        [padCoordNums[2], padCoordNums[3]],
                        [0, padCoordNums[4]],
                        [-padCoordNums[2], padCoordNums[3]],
                        [-padCoordNums[0], padCoordNums[1]]]
        
        padCoords2 = [[x[0], -x[1]] for x in padCoords][::-1]

        poly1 = Polygon(padCoords).buffer(-0.04, join_style=self.join_style, quad_segs=self.quad_segs)
        poly1 = poly1.buffer(0.04, join_style=self.join_style, quad_segs=self.quad_segs)

        poly2 = Polygon(padCoords2).buffer(-0.04, join_style=self.join_style, quad_segs=self.quad_segs)
        poly2 = poly2.buffer(0.04, join_style=self.join_style, quad_segs=self.quad_segs)

        return poly1, poly2
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from typing import Tuple
    import numpy as np

    # Your class definitions would be here (PolygonConstructor + SymmetricTransmonPolygonConstructor)

    def plot_polygons(polygon_pair: Tuple[Polygon, Polygon], ax, label: str, color: str):
        for poly in polygon_pair:
            if not poly.is_empty:
                x, y = poly.exterior.xy
                ax.plot(x, y, label=label, color=color)

    # Instantiate the constructor
    constructor = SymmetricTransmonPolygonConstructor()

    # Width values to visualize
    widths = [0.04, 0.06, 0.08]

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['blue', 'green', 'red']

    for width, color in zip(widths, colors):
        params = np.array([width])
        polys = constructor.make_polygons(params)
        plot_polygons(polys, ax, label=f'width={width}', color=color)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Symmetric Transmon Polygons at Different Widths')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
