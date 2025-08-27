import pytest
import math
import numpy as np

from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf.AdjointSolver import Optimiser, AdjointEvaluator

@pytest.fixture(scope="module")
def params():
    return np.array([1.19])

@pytest.fixture(scope="module")
def perturbation():
    return np.array([0.01])

@pytest.fixture(scope="module")
def parametric_designer():
    return SymmetricTransmonDesign()

@pytest.fixture(scope="module")
def adjoint_evaluater(parametric_designer):
    return AdjointEvaluator(parametric_designer)

@pytest.fixture(scope="module")
def design(parametric_designer, params):
    return parametric_designer.build_qk_design(params)

def test_fwd_calc(adjoint_evaluater, design):
    adjoint_evaluater._fwd_calculation(design)

def test_adj_calc(adjoint_evaluater, design):
    adjoint_evaluater._adjoint_calculation(design, 1)

@pytest.fixture(scope='module')
def boundary_inner_product(params, perturbation, adjoint_evaluater, design):
    evaluator = adjoint_evaluater
    fwd_sparams = evaluator._fwd_calculation(design)
    adjoint_strength = evaluator._adjoint_strength(
        fwd_sparams, evaluator.adjoint_source_location
    )
    adj_sparams = evaluator._adjoint_calculation(design, adjoint_strength)
    value = evaluator._calc_adjoint_forward_product(
        params, perturbation, fwd_sparams, adj_sparams
    )
    return value

def test_inner_product_runs(boundary_inner_product):
    assert boundary_inner_product is not None

def test_inner_product_real(boundary_inner_product):
    realness_ratio = abs(boundary_inner_product) / abs(boundary_inner_product.real)
    assert 0.9 < realness_ratio and 1.1 > realness_ratio
