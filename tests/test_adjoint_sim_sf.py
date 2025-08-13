# from adjoint_sim_sf.AdjointSolver import AdjointOptimiser
from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from adjoint_sim_sf.Simulation import SimulationRunner
from unittest.mock import MagicMock, patch
import numpy as np


# def test_adjointoptimiser_fwd_and_adjoint_calculations_run():
#     """Ensure AdjointOptimiser forward and adjoint methods run without error."""
#     test_width = np.array([0.2])
#     builder = SymmetricTransmonBuilder()
#     optimiser = AdjointOptimiser(test_width, lr=1e-3, qiskit_builder=builder)

#     # build design and run forward sim
#     optimiser.design = builder.get_design(test_width)
#     fwd_result = optimiser._fwd_calculation()
#     assert callable(fwd_result)

#     # store dummy field data for adjoint simyou 
#     optimiser.fwd_field_data = {
#         "coords": fwd_result('E')[0]  # example: coords from E-field output
#     }

#     optimiser._adjoint_calculation()  # Should not raise error


# def test_symmetic_transmon_design_initialization():
#     """Check that SymmetricTransmonDesign can be initialized."""
#     design = SymmetricTransmonDesign()
#     assert design is not None, "SymmetricTransmonDesign should be initializable"


# def test_simulation_runner_initialization():
#     """Check that SimulationRunner can be initialized."""
#     runner = SimulationRunner(freq_value=5e9)
#     assert runner is not None, "SimulationRunner should be initializable"
#     assert runner.freq_value == 5e9


# def test_symmetric_transmon_build_design():
#     """Ensure SymmetricTransmonDesign's build_design method runs."""
#     design = SymmetricTransmonDesign()
#     # Mock the builder to avoid Qiskit Metal dependency details
#     design.design_builder.get_design = MagicMock(return_value="mock_design")
#     built_design = design.build_design(parameters=np.array([0.2]))
#     assert built_design == "mock_design"


# @patch('adjoint_sim_sf.Simulation.COMSOL_Model')
# def test_simulation_runner_run_forward(mock_comsol_model):
#     """Ensure SimulationRunner's run_forward method can be called."""
#     mock_sim_instance = MagicMock()
#     mock_comsol_model.return_value.initialize_model.return_value = None
#     mock_comsol_model.return_value.run.return_value = mock_sim_instance

#     runner = SimulationRunner(freq_value=5e9)
#     # The design and source location can be simple mocks for this test
#     result = runner.run_forward(design="mock_design", source_location=[0, 0])
#     assert result is not None


# @patch('adjoint_sim_sf.Simulation.COMSOL_Model')
# def test_simulation_runner_run_adjoint(mock_comsol_model):
#     """Ensure SimulationRunner's run_adjoint method can be called."""
#     mock_sim_instance = MagicMock()
#     mock_comsol_model.return_value.initialize_model.return_value = None
#     mock_comsol_model.return_value.run.return_value = mock_sim_instance

#     runner = SimulationRunner(freq_value=5e9)
#     result = runner.run_adjoint(design="mock_design", source_location=[0, 0], source_strength=1.0)
#     assert result is not None
