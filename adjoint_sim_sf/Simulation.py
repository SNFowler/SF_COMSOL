from datetime import datetime
import pathlib

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

    def run_forward(self, design, source_locations, source_strength=20.0):
        """Run the forward simulation."""
        dipole_moment_vecdir =  [0, 1, 0]
        return self._run_sim("fwdmodel", design, source_locations, source_strength, dipole_moment_vecdir)

    def run_adjoint(self, design, source_locations, source_strength):
        """Run the adjoint simulation."""
        dipole_moment_vecdir =  [0, 1, 0]
        return self._run_sim("adjmodel", design, source_locations, source_strength, dipole_moment_vecdir)

    def eval_field_at_pts(self, sParams: COMSOL_Simulation_RFsParameters, expr, points, freq_index=1):
        return sParams.eval_field_at_pts(expr, points, freq_index)

    def eval_fields_over_mesh(self, sParams: COMSOL_Simulation_RFsParameters):
        """Evaluate all fields over the simulation mesh."""
        return sParams.eval_fields_over_mesh()

    def save(self):
        if not self.latest_cmsl:
            raise RuntimeError("Simulation Runner has no latest cmsl to save.")
        ts = datetime.now().strftime("%Y%m%d%H%M%S-%f")
        self.latest_cmsl.save((pathlib.Path(__file__).resolve().parent / "comsol_output") / ts)

    def _unit_conversion(coords: np.ndarray, scale = 1e-6):
        return coords * scale


    def _run_sim(self, name, design, src_locations_vec, dipole_moment_p, dipole_moment_vecdir):
        """Internal method to set up and run a COMSOL simulation.
        
        @TODO: dipole_moment_vecdir: unused
        @TODO: Let this function accept multiple dipole moments, multiple sources and potentially multiple dipole_moments
        """
        cmsl = COMSOL_Model(name)
        sim = COMSOL_Simulation_RFsParameters(cmsl, adaptive='None')

        # build design in simulation
        cmsl.initialize_model(design, [sim], bottom_grounded=True)
        cmsl.add_metallic(1, threshold=1e-12, fuse_threshold=1e-10)
        cmsl.add_ground_plane()
        cmsl.fuse_all_metals()
        sim.create_port_JosephsonJunction('junction', L_J=4.3e-9, C_J=10e-15, R_J=10e3)

        # add source

        for src_location in src_locations_vec:
            sim.add_electric_point_dipole(src_location, dipole_moment_p, dipole_moment_vecdir)

        # create mesh
        cmsl.fine_mesh_around_comp_boundaries(['pad1', 'pad2'],
                                              minElementSize=10e-6,
                                              maxElementSize=50e-6)
        cmsl.build_geom_mater_elec_mesh(skip_meshing=True, mesh_structure='Fine')

        # run simulatiom
        sim.set_freq_values([self.freq_value])
        sim.run()
        
        self.latest_cmsl = cmsl
        self.latest_sim =  sim

        return sim
