import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from adjoint_sim_sf import AdjointOptimiser, DesignBuilder, PolygonConstructor, SymmetricTransmonPolygonConstructor

class DummyDesignBuilder:
    def rebuild(self): pass  # placeholder for now

@pytest.fixture
def optimiser():
    initial_params = np.array([0.05])
    lr = 0.01
    polygon_constructor = SymmetricTransmonPolygonConstructor()
    dummy_design = DummyDesignBuilder()

    return AdjointOptimiser(initial_params, lr, dummy_design, polygon_constructor)

def test_compute_Ap_returns_shapely_objects(optimiser):
    result = optimiser._compute_Ap()
    
    assert isinstance(result, list)
    assert len(result) == len(optimiser.current_params)
    for item in result:
        assert isinstance(item, (BaseGeometry, Polygon))  # shapely geometry
