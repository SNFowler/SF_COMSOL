import numpy as np
from SQDMetal.COMSOL.Model import COMSOL_Model
from SQDMetal.COMSOL.SimRFsParameter import COMSOL_Simulation_RFsParameters

class SimulationRunner:
    """
    Provides an interface for the AdjointOptimiser to run COMSOL simulations
    and query field data.
    """
    def __init__(self, freq_value):
        self.freq_value = freq_value

        self.latest_cmsl = None
        self.latest_sim = None

    def run_forward(self, design, source_location, source_strength=1.0):
        """Run the forward simulation."""
        return self._run_sim("fwdmodel", design, source_location, source_strength)

    def run_adjoint(self, design, source_location, source_strength):
        """Run the adjoint simulation."""
        return self._run_sim("adjmodel", design, source_location, source_strength)

    def eval_field_at_pts(self, sParams, points, field = "E"):
        """Evaluate the specified field at given points."""
        return sParams.eval_field_at_pts(field, points)

    def eval_fields_over_mesh(self, sParams):
        """Evaluate all fields over the simulation mesh."""
        return sParams.eval_fields_over_mesh()

    def _run_sim(self, name, design, source_location, source_strength):
        """Internal method to set up and run a COMSOL simulation."""
        cmsl = COMSOL_Model(name)
        sim = COMSOL_Simulation_RFsParameters(cmsl, adaptive='None')
        cmsl.initialize_model(design, [sim], bottom_grounded=True)
        cmsl.add_metallic(1, threshold=1e-12, fuse_threshold=1e-10)
        cmsl.add_ground_plane()
        cmsl.fuse_all_metals()
        sim.create_port_JosephsonJunction('junction', L_J=4.3e-9, C_J=10e-15, R_J=10e3)
        sim.add_electric_point_dipole(source_location, source_strength, [0, 1, 0])
        cmsl.fine_mesh_around_comp_boundaries(['pad1', 'pad2'],
                                              minElementSize=10e-6,
                                              maxElementSize=50e-6)
        cmsl.build_geom_mater_elec_mesh(skip_meshing=True, mesh_structure='Fine')
        sim.set_freq_values([self.freq_value])
        sim.run()
        
        self.latest_cmsl = cmsl
        self.latest_sim =  sim

        return sim
