from adjoint_sim_sf.DesignBuilder import DesignBuilder, SymmetricTransmonBuilder
from adjoint_sim_sf.AdjointSolver import AdjointOptimiser
import numpy as np


def test_adjointoptimiser_fwd_and_adjoint_calculations_run():
    """Ensure AdjointOptimiser forward and adjoint methods run without error."""
    test_width = np.array([0.2])
    builder = SymmetricTransmonBuilder()
    optimiser = AdjointOptimiser(test_width, lr=1e-3, qiskit_builder=builder)

    # build design and run forward sim
    optimiser.design = builder.get_design(test_width)
    fwd_result = optimiser._fwd_calculation()
    assert callable(fwd_result)

    # store dummy field data for adjoint simyou 
    optimiser.fwd_field_data = {
        "coords": fwd_result('E')[0]  # example: coords from E-field output
    }

    optimiser._adjoint_calculation()  # Should not raise error
