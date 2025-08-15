import numpy as np
import shapely
from shapely.geometry import Polygon
import types
import pytest

from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign, SymmetricTransmonPolygonConstructor

class FakeBuilder:
    """Tiny stub to avoid qiskit_metal in tests."""
    def __init__(self):
        self.last_polys = None
        self.return_token = object()

    def get_design(self, shapely_components):
        # record what we were asked to build
        self.last_polys = shapely_components
        # return a harmless token so tests can assert identity
        return self.return_token

def test_geometry_returns_two_polygons():
    design = SymmetricTransmonDesign()
    polys = design.geometry(np.array([0.06], dtype=float))
    assert isinstance(polys, tuple) and len(polys) == 2
    assert all(isinstance(p, Polygon) for p in polys)
    # Non-empty and valid
    assert all((not p.is_empty) and p.is_valid for p in polys)

def test_geometry_changes_with_param():
    design = SymmetricTransmonDesign()
    p_small = design.geometry(np.array([0.05], dtype=float))
    p_large = design.geometry(np.array([0.08], dtype=float))
    # area should increase with width for at least one pad
    area_small = sum(pi.area for pi in p_small)
    area_large = sum(pi.area for pi in p_large)
    assert area_large > area_small

def test_compute_Ap_nonzero_when_param_perturbed():
    design = SymmetricTransmonDesign()
    base_params = np.array([0.06], dtype=float)
    ap = design.compute_Ap(base_params, perturbation=np.array([1e-3], dtype=float))
    # A_p is a geometric difference; should be non-empty for nonzero perturbation
    assert hasattr(ap, "is_empty")
    assert not ap.is_empty

def test_compute_Ap_zero_when_zero_perturbation():
    design = SymmetricTransmonDesign()
    base_params = np.array([0.06], dtype=float)
    ap = design.compute_Ap(base_params, perturbation=np.array([0.0], dtype=float))
    # With zero perturbation, the symmetric difference should be empty (or near-empty)
    # Some kernels may return a GeometryCollection([]); treat empty as acceptable.
    assert ap.is_empty

def test_build_design_calls_builder_with_polys(monkeypatch):
    design = SymmetricTransmonDesign()
    # Swap out the real builder for a stub so we avoid qiskit_metal
    fake = FakeBuilder()
    design.design_builder = fake

    params = np.array([0.06], dtype=float)
    token = design.build_qk_design(params)

    # Returned object is exactly whatever the builder returned
    assert token is fake.return_token
    # And the builder received two polygons from the constructor
    assert isinstance(fake.last_polys, tuple) and len(fake.last_polys) == 2
    assert all(hasattr(p, "area") for p in fake.last_polys)

def test_polygon_constructor_basic_shape_validity():
    ctor = SymmetricTransmonPolygonConstructor()
    p1, p2 = ctor.make_polygons(np.array([0.06], dtype=float))
    assert p1.is_valid and p2.is_valid
    assert p1.buffer(0).is_valid and p2.buffer(0).is_valid  # catch subtle self-intersections

def test_polygon_constructor_monotonicity_basic():
    """Lo-fi monotonicity sanity check: bigger width -> bigger total area."""
    ctor = SymmetricTransmonPolygonConstructor()
    a_small = sum(p.area for p in ctor.make_polygons(np.array([0.05], dtype=float)))
    a_large = sum(p.area for p in ctor.make_polygons(np.array([0.09], dtype=float)))
    assert a_large > a_small

def test_show_polygons_runs(monkeypatch):
    import matplotlib.pyplot as plt
    # Prevent actual window popping up
    monkeypatch.setattr(plt, "show", lambda: None)

    design = SymmetricTransmonDesign()
    design.show_polygons(np.array([0.1]))  # or whatever params make sense

