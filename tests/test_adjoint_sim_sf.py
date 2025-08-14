from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf.Simulation import SimulationRunner
from adjoint_sim_sf.AdjointSolver import Optimiser, AdjointEvaluator

import numpy as np
import pytest
from adjoint_sim_sf import AdjointOptimiser, SymmetricTransmonBuilder, SymmetricTransmonPolygonConstructor

def test_basic_fwd_and_rev_sim_run():
    """Tests that the forward and reverse simulations can be run."""
    initial_params = np.array([0.19971691])
    polygon_constructor = SymmetricTransmonPolygonConstructor()
    real_design = SymmetricTransmonBuilder()

    optimiser = AdjointOptimiser(
        initial_params, 0.01, real_design, polygon_constructor
    )

    optimiser._update_qiskit_design()
    optimiser._fwd_calculation()
    optimiser._adjoint_calculation()

    assert hasattr(optimiser.fwd_field_sParams, "eval_field_at_pts")
    assert callable(optimiser.fwd_field_sParams.eval_field_at_pts)
    assert hasattr(optimiser.adjoint_field_sParams, "eval_field_at_pts")
    assert callable(optimiser.adjoint_field_sParams.eval_field_at_pts)
