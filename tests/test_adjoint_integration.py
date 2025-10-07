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
def simulation_results(adjoint_evaluator, design):
    """
    Run forward and adjoint simulations once for reuse in tests.
    FIX: _adjoint_calculation must receive fwd_sparams (not strengths).
    """
    fwd_sparams = adjoint_evaluator._fwd_calculation(design)
    adj_sparams = adjoint_evaluator._adjoint_calculation(design, fwd_sparams)
    return fwd_sparams, adj_sparams

# --- Tests
def test_fwd_calc(adjoint_evaluator, design):
    out = adjoint_evaluator._fwd_calculation(design)
    assert hasattr(out, "eval_field_at_pts")

def test_adj_calc(adjoint_evaluator, design):
    # FIX: pass a real forward sim result
    fwd = adjoint_evaluator._fwd_calculation(design)
    adj = adjoint_evaluator._adjoint_calculation(design, fwd)
    assert hasattr(adj, "eval_field_at_pts")

def test_boundary_inner_product(params, single_perturbation, adjoint_evaluator, simulation_results):
    """Test the boundary inner product calculation."""
    fwd_sparams, adj_sparams = simulation_results
    inner_product = adjoint_evaluator.compute_boundary_inner_product(
        params, single_perturbation, fwd_sparams, adj_sparams
    )
    assert inner_product is not None
    assert isinstance(inner_product, (complex, np.complexfloating, np.ndarray))

def test_evaluate(params, single_perturbation, adjoint_evaluator):
    """Test the full evaluate method returns gradient and loss."""
    grad, loss = adjoint_evaluator.evaluate(params, single_perturbation, verbose=False)
    assert grad is not None and np.isfinite(grad).all()
    assert loss is not None and np.isfinite(loss).all()
    assert np.real(loss).min() >= 0  # allow array loss
