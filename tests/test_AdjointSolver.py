import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from adjoint_sim_sf import AdjointOptimiser, SymmetricTransmonBuilder, PolygonConstructor, SymmetricTransmonPolygonConstructor

@pytest.fixture
def optimiser():
    initial_params = np.array([0.05])
    lr = 0.01
    polygon_constructor = SymmetricTransmonPolygonConstructor()
    real_design = SymmetricTransmonBuilder() 

    return AdjointOptimiser(initial_params, lr, real_design, polygon_constructor)


def test_compute_Ap_returns_shapely_objects(optimiser):
    result = optimiser._compute_Ap()
    assert isinstance(result, list)
    assert len(result) == len(optimiser.current_params)
    for item in result:
        assert isinstance(item, (BaseGeometry, Polygon))

def test_fwd_sim(optimiser):
    optimiser._update_qiskit_design() 
    sim = optimiser._fwd_calculation()
    assert hasattr(optimiser.fwd_field_sParams, "eval_field_at_pts")
    assert callable(optimiser.fwd_field_sParams.eval_field_at_pts)

def test_fwd_and_rev_sim(optimiser):
    optimiser._update_qiskit_design()
    optimiser._fwd_calculation()
    sim = optimiser._adjoint_calculation()
    assert hasattr(optimiser.adjoint_field_sParams, "eval_field_at_pts")
    assert callable(optimiser.adjoint_field_sParams.eval_field_at_pts)

def test_full_gradient_computation_runs(optimiser):
    optimiser._update_qiskit_design()
    optimiser._fwd_calculation()
    optimiser._adjoint_calculation()
    optimiser._extract_E_fields()
    optimiser._compute_gradient()

    assert optimiser.grad is not None
    assert isinstance(optimiser.grad, np.ndarray)
    assert optimiser.grad.shape == (len(optimiser.current_params),)

def test_adjoint_method_sweep(optimiser):
    print("Commencing test")
    baseline_param = 0.19971691
    sweep_vals = np.linspace(-0.05, 0.05, 3) + baseline_param

    print("commencing sweep")
    gradients = optimiser._compute_adjoint_gradient_sweep(sweep_vals, verbose=True)

    assert gradients is not None
    assert isinstance(gradients, np.ndarray)
    