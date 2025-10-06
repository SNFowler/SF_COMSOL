from datetime import datetime
import pathlib

from dataclasses import dataclass

import numpy as np
from SQDMetal.COMSOL.Model import COMSOL_Model
from SQDMetal.COMSOL.SimRFsParameter import COMSOL_Simulation_RFsParameters

from .Sources import Source
from typing import List

#TODO update run_foward, run_adjoint, _run_sim to use List[Source] 
class SimulationRunner:
    """
    Provides an interface for the AdjointOptimiser to run COMSOL simulations
    and query field data.
    """
    def __init__(self, freq_value):
        self.freq_value = freq_value

        self.latest_cmsl = None
        self.latest_sim = None

    def run_forward(self, design, sources: List[Source]):
        """Run the forward simulation with multiple sources."""
        return self._run_sim("fwdmodel", design, sources)

    def run_adjoint(self, design, sources: List[Source]):
        """Run the adjoint simulation with multiple sources."""
        return self._run_sim("adjmodel", design, sources)

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


    def _run_sim(self, name, design, sources: List[Source]):
        """Internal method to set up and run a COMSOL simulation."""
        cmsl = COMSOL_Model(name)
        sim = COMSOL_Simulation_RFsParameters(cmsl, adaptive='None')
        
        # build design in simulation
        cmsl.initialize_model(design, [sim], bottom_grounded=True)
        cmsl.add_metallic(1, threshold=1e-12, fuse_threshold=1e-10)
        cmsl.add_ground_plane()
        cmsl.fuse_all_metals()
        sim.create_port_JosephsonJunction('junction', L_J=4.3e-9, C_J=10e-15, R_J=10e3)
        
        # add sources
        for src in sources:
            sim.add_electric_point_dipole(src.location, src.strength, src.direction)
        
        # create mesh and run
        cmsl.fine_mesh_around_comp_boundaries(['pad1', 'pad2'],
                                            minElementSize=10e-6,
                                            maxElementSize=50e-6)
        cmsl.build_geom_mater_elec_mesh(skip_meshing=True, mesh_structure='Fine')
        sim.set_freq_values([self.freq_value])
        sim.run()
        
        self.latest_cmsl = cmsl
        self.latest_sim = sim
        return sim
