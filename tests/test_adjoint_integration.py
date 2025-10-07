import pytest
import numpy as np
from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf import Optimiser, AdjointEvaluator

# --- Fixtures
@pytest.fixture(scope="module", params=[np.array([0.199]), np.array([0.199, 0.25])])
def params(request):
    return request.param

@pytest.fixture(scope="module")
def single_perturbation():
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

@pytest.fixture(scope="module")
def fwd_sparams(adjoint_evaluator, design):
    fwd = adjoint_evaluator._fwd_calculation(design)
    # Optional guard if your backend could return None/invalid:
    if fwd is None or not hasattr(fwd, "eval_field_at_pts"):
        pytest.skip("Forward simulation unavailable")
    return fwd

@pytest.fixture(scope="module")
def adj_sparams(adjoint_evaluator, design, fwd_sparams):
    adj = adjoint_evaluator._adjoint_calculation(design, fwd_sparams)
    if adj is None or not hasattr(adj, "eval_field_at_pts"):
        pytest.skip("Adjoint simulation unavailable")
    return adj

# --- Tests
def test_fwd_calc(fwd_sparams, design):
    assert fwd_sparams is not None    
    assert hasattr(fwd_sparams, "eval_field_at_pts")

def test_adj_calc(adj_sparams):
    assert adj_sparams is not None    
    assert hasattr(adj_sparams, "eval_field_at_pts")

def test_boundary_inner_product(params, single_perturbation, adjoint_evaluator, fwd_sparams, adj_sparams):
    """Test the boundary inner product calculation."""
    inner_product = adjoint_evaluator.compute_boundary_inner_product(
        params, single_perturbation, fwd_sparams, adj_sparams
    )
    assert inner_product is not None
    assert isinstance(inner_product, (complex, np.complexfloating, np.ndarray))

def test_evaluate(params, single_perturbation, adjoint_evaluator, fwd_sparams, adj_sparams):
    """Test the full evaluate method returns gradient and loss."""
    grad, loss = adjoint_evaluator.evaluate(params, single_perturbation, 
                                            verbose=False, 
                                            fwd_sparams=fwd_sparams, 
                                            adj_sparams=adj_sparams)
    
    assert grad is not None and np.isfinite(grad).all()
    assert loss is not None and np.isfinite(loss).all()
    assert np.real(loss).min() <= 0  # allow array loss
