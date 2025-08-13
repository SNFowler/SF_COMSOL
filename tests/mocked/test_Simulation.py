# tests/test_simulation_runner.py
import numpy as np
import types
import pytest

# import the module under test
# adjust the import path to wherever SimulationRunner lives
from adjoint_sim_sf.Simulation import SimulationRunner

class FakeSim:
    def __init__(self, model_ref, adaptive='None'):
        self.model_ref = model_ref
        self.created_jj = None
        self.dipole = None
        self.freqs = None
        self.ran = False

    # API mirrored from real sim object
    def create_port_JosephsonJunction(self, name, L_J, C_J, R_J):
        self.created_jj = (name, L_J, C_J, R_J)

    def add_electric_point_dipole(self, loc, strength, pol):
        self.dipole = (tuple(loc), float(strength), tuple(pol))

    def set_freq_values(self, freqs):
        self.freqs = tuple(freqs)

    def run(self):
        self.ran = True

    # used by SimulationRunner passthroughs
    def eval_field_at_pts(self, field, points):
        return np.ones((len(points), 3))  # dummy vector field

    def eval_fields_over_mesh(self):
        # coords: N x 3, E: N x 3
        coords = np.array([[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0]])
        E = np.zeros_like(coords)
        return {"coords": coords, "E": E}

class FakeModel:
    def __init__(self, name):
        self.name = name
        self.init_args = None
        self.metallic = None
        self.gp = False
        self.fused = False
        self.meshed = None
        self.mesh_call = None

    def initialize_model(self, design, sims, bottom_grounded=True):
        # Just record params
        self.init_args = dict(design=design, sims=list(sims), bottom_grounded=bottom_grounded)

    def add_metallic(self, *args, **kwargs):
        self.metallic = (args, kwargs)

    def add_ground_plane(self):
        self.gp = True

    def fuse_all_metals(self):
        self.fused = True

    def fine_mesh_around_comp_boundaries(self, comps, minElementSize, maxElementSize):
        self.mesh_call = dict(comps=tuple(comps),
                              min=minElementSize, max=maxElementSize)

    def build_geom_mater_elec_mesh(self, skip_meshing=True, mesh_structure='Fine'):
        self.meshed = dict(skip=skip_meshing, structure=mesh_structure)

@pytest.fixture
def patched_comsol(monkeypatch):
    # patch the names inside the module under test
    import adjoint_sim_sf.Simulation as sim_mod
    monkeypatch.setattr(sim_mod, "COMSOL_Model", FakeModel, raising=True)
    monkeypatch.setattr(sim_mod, "COMSOL_Simulation_RFsParameters", FakeSim, raising=True)
    return sim_mod

def test_run_forward_wires_calls(patched_comsol):
    runner = SimulationRunner(freq_value=8.0333e9)

    fake_design = object()  # we don't care what it is, just pass-through
    src_loc = [1e-3, 2e-3, 3e-6]

    sim = runner.run_forward(fake_design, src_loc, source_strength=2.5)

    # assertions on FakeModel + FakeSim side effects
    assert isinstance(sim, FakeSim)
    model = sim.model_ref
    assert isinstance(model, FakeModel)
    # initialize_model recorded design and sims
    assert model.init_args["design"] is fake_design
    assert model.init_args["bottom_grounded"] is True
    # metallic setup done
    assert model.metallic is not None
    assert model.gp is True
    assert model.fused is True
    # meshing configuration captured
    assert model.meshed["structure"] == "Fine"
    assert model.mesh_call["comps"] == ("pad1", "pad2")
    # sim settings
    assert sim.created_jj[0] == "junction"
    assert sim.dipole == (tuple(src_loc), 2.5, (0, 1, 0))
    assert sim.freqs == (runner.freq_value,)
    assert sim.ran is True

def test_run_adjoint_uses_different_model_name_and_strength(patched_comsol):
    runner = SimulationRunner(freq_value=7.0e9)
    sim = runner.run_adjoint(object(), [0, 0, 0], source_strength=0.123)
    assert isinstance(sim, FakeSim)
    # strength propagated
    assert sim.dipole[1] == 0.123
    # frequency set
    assert sim.freqs == (7.0e9,)

def test_eval_helpers_delegate(patched_comsol):
    runner = SimulationRunner(freq_value=1.0)
    sim = runner._run_sim("any", object(), [0, 0, 0], 1.0)
    vals = runner.eval_field_at_pts(sim, "E", np.array([[0, 0, 0], [1, 0, 0]]))
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (2, 3)
    mesh = runner.eval_fields_over_mesh(sim)
    assert set(mesh.keys()) == {"coords", "E"}
    assert mesh["coords"].shape == mesh["E"].shape
