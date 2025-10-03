import pytest
import math
import numpy as np

from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf.AdjointSolver import Optimiser, AdjointEvaluator

# --- Fixtures

@pytest.fixture(scope="module")
def params():
    return np.array([0.199])

@pytest.fixture(scope="module")
def perturbation():
    return np.array([0.01])

@pytest.fixture(scope="module")
def parametric_designer():
    return SymmetricTransmonDesign()

@pytest.fixture(scope="module")
def adjoint_evaluator(parametric_designer):
    return AdjointEvaluator(parametric_designer)

@pytest.fixture(scope="module")
def design(parametric_designer, params):
    return parametric_designer.build_qk_design(params)

@pytest.fixture(scope='module')
def boundary_inner_product(params, perturbation, adjoint_evaluator, design):
    evaluator = adjoint_evaluator
    
    # Run forward simulation
    fwd_sparams = evaluator._fwd_calculation(design)
    
    # Calculate adjoint strength and run adjoint simulation
    adjoint_strength = evaluator._adjoint_strength(
        fwd_sparams, evaluator.adjoint_source_location
    )
    adj_sparams = evaluator._adjoint_calculation(design, adjoint_strength)
    
    # MISSING: Compute boundary velocity field and reference coordinates
    boundary_velocity_field, reference_coord, _ = \
        evaluator.parametric_designer.compute_boundary_velocity(params, perturbation)
    
    # Now call with correct arguments
    value = evaluator._calc_adjoint_forward_product(
        boundary_velocity_field, reference_coord, fwd_sparams, adj_sparams,
        adjoint_rotation=evaluator.adjoint_rotation
    )
    return value

# --- Tests

def test_fwd_calc(adjoint_evaluator, design):
    adjoint_evaluator._fwd_calculation(design)


def test_adj_calc(adjoint_evaluator, design):
    adjoint_evaluator._adjoint_calculation(design, 1)


def test_inner_product_runs(boundary_inner_product):
    assert boundary_inner_product is not None
