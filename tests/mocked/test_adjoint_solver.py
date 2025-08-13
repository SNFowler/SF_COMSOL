import numpy as np
import pytest

import adjoint_sim_sf.AdjointSolver as solver_mod


# ---------- compact fakes ----------
class _FakeDesign:
    def rebuild(self): pass

class _FakeParametricDesign:
    def __init__(self):
        self.A_p = object()
        self.ap_weight_calls = 0
        self.compute_Ap_calls = 0
    def build_design(self, params): return _FakeDesign()
    def geometry(self, params): return ()
    def compute_Ap(self, params, perturb):
        self.compute_Ap_calls += 1
        return self.A_p
    def ap_weight(self, point, ap_obj):
        assert ap_obj is self.A_p
        self.ap_weight_calls += 1
        return 1.0

class _FakeSim:
    def eval_field_at_pts(self, field, points): return np.ones((len(points), 3))
    def eval_fields_over_mesh(self):
        coords = np.array([[0,0,0],[1,0,0]], dtype=float)
        E = E = np.ones_like(coords, dtype=float)
        return {"coords": coords, "E": E}
    def run(self): pass

class _FakeRunner:
    def __init__(self, freq): self.n_fwd = 0; self.n_adj = 0
    def run_forward(self, design, loc, strength=1.0): self.n_fwd += 1; return _FakeSim()
    def run_adjoint(self, design, loc, strength): self.n_adj += 1; return _FakeSim()
    def eval_field_at_pts(self, sim, field, pts): return sim.eval_field_at_pts(field, pts)
    def eval_fields_over_mesh(self, sim): return sim.eval_fields_over_mesh()


# ---------- shared monkeypatch ----------
@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    # Make __init__ happy
    class _FakeCOMSOLModel: _engine = object()
    monkeypatch.setattr(solver_mod, "COMSOL_Model", _FakeCOMSOLModel, raising=True)
    # Swap SimulationRunner class
    monkeypatch.setattr(solver_mod, "SimulationRunner", _FakeRunner, raising=True)


# ---------- tests ----------
def test_evaluate_basic(monkeypatch):
    design = _FakeParametricDesign()
    ev = solver_mod.AdjointEvaluator(parametric_designer=design, perturb=1e-5)

    loss, grad = ev.evaluate(np.array([0.06]), verbose=False)

    assert loss > 0
    assert isinstance(grad, np.ndarray) and grad.shape == (1,)
    assert design.compute_Ap_calls == 1
    # two mesh points -> at least 2 ap_weight calls
    assert design.ap_weight_calls >= 2

def test_optimiser_step_updates_params():
    design = _FakeParametricDesign()
    ev = solver_mod.AdjointEvaluator(parametric_designer=design, perturb=1e-5)
    opt = solver_mod.Optimiser(initial_params=np.array([0.06]), lr=0.1, evaluator=ev)

    before = opt.params.copy()
    loss, grad = opt.step()
    assert not np.allclose(opt.params, before)
    assert opt.history and isinstance(opt.history[-1][1], float)

def test_calls_happen_once_each(monkeypatch):
    design = _FakeParametricDesign()
    ev = solver_mod.AdjointEvaluator(parametric_designer=design, perturb=1e-5)

    # Spy on the instance methods with counters
    fwd_calls = {"n": 0}; adj_calls = {"n": 0}
    orig_fwd = ev.sim.run_forward; orig_adj = ev.sim.run_adjoint

    monkeypatch.setattr(ev.sim, "run_forward",
        lambda *a, **k: (fwd_calls.__setitem__("n", fwd_calls["n"]+1) or orig_fwd(*a, **k)))
    monkeypatch.setattr(ev.sim, "run_adjoint",
        lambda *a, **k: (adj_calls.__setitem__("n", adj_calls["n"]+1) or orig_adj(*a, **k)))

    ev.evaluate(np.array([0.06]))
    assert fwd_calls["n"] == 1
    assert adj_calls["n"] == 1
