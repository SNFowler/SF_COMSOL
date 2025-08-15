import numpy as np
import pytest

from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf.AdjointSolver import Optimiser, AdjointEvaluator

def test_fwd_calc():
    params = np.array([1.19])
    parametric_designer = SymmetricTransmonDesign()
    adjoint_evaluater = AdjointEvaluator(parametric_designer)

    design = parametric_designer.build_qk_design(params)


    adjoint_evaluater._fwd_calculation(design)

def test_adj_calc():
    params = np.array([1.19])
    parametric_designer = SymmetricTransmonDesign()
    adjoint_evaluater = AdjointEvaluator(parametric_designer)

    design = parametric_designer.build_qk_design(params)


    adjoint_evaluater._adjoint_calculation(design, 1)