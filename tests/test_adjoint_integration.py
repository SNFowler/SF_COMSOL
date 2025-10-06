import pytest
import math
import numpy as np
from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf import Optimiser, AdjointEvaluator

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
def simulation_results(adjoint_evaluator, design):
    """Run forward and adjoint simulations once for reuse in tests."""
    evaluator = adjoint_evaluator
    
    # Run forward simulation
    fwd_sparams = evaluator._fwd_calculation(design)
    
    # Calculate adjoint strength and run adjoint simulation
    adjoint_strength = evaluator._adjoint_strength(
        fwd_sparams, evaluator.adjoint_source_locations
    )
    adj_sparams = evaluator._adjoint_calculation(design, adjoint_strength)
    
    return fwd_sparams, adj_sparams

# --- Tests
def test_fwd_calc(adjoint_evaluator, design):
    adjoint_evaluator._fwd_calculation(design)

def test_adj_calc(adjoint_evaluator, design):
    adjoint_evaluator._adjoint_calculation(design, 1)

def test_boundary_inner_product(params, perturbation, adjoint_evaluator, simulation_results):
    """Test the boundary inner product calculation."""
    fwd_sparams, adj_sparams = simulation_results
    
    inner_product = adjoint_evaluator.compute_boundary_inner_product(
        params, perturbation, fwd_sparams, adj_sparams
    )
    
    assert inner_product is not None
    assert isinstance(inner_product, (complex, np.complexfloating, np.ndarray))

def test_evaluate(params, perturbation, adjoint_evaluator):
    """Test the full evaluate method returns gradient and loss."""
    grad, loss = adjoint_evaluator.evaluate(params, perturbation, verbose=False)
    
    # Check gradient exists and is numeric
    assert grad is not None
    assert np.isfinite(grad).all()
    
    # Check loss exists and is positive (field magnitude squared)
    assert loss is not None
    assert np.isfinite(loss).all()
    assert np.real(loss) >= 0  # Loss should be non-negative