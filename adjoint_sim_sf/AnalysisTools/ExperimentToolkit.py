"""Minimal helpers for running COMSOL adjoint experiments from Python."""
from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from adjoint_sim_sf.AdjointSolver import AdjointEvaluator, Optimiser
from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign
from SQDMetal.COMSOL.Model import COMSOL_Model


@dataclass(frozen=True)
class SweepSettings:
    center: float = 0.199
    width: float = 0.035
    num: int = 21
    reuse_fields: bool = False
    angles: Sequence[float] = (0.0,)
    adjoint_rotation: Optional[float] = None
    filename_stem: str = "sweep"


@dataclass(frozen=True)
class GradientDescentSettings:
    initial_param: Sequence[float]
    steps: int = 8
    lr: float = 0.01
    perturbation: Optional[Sequence[float]] = None
    filename_stem: str = "gradient_descent"


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    base_params: Sequence[float]
    perturbation: Sequence[float] = (1e-5,)
    sweep: SweepSettings = field(default_factory=SweepSettings)
    source_locations: Optional[Sequence[Sequence[float]]] = None
    source_strength: float = 1e-2
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)
    gradient_descent: Optional[GradientDescentSettings] = None
    resume: bool = True


@dataclass
class ExperimentResult:
    variant: ExperimentVariant
    run_dir: Path
    outputs: List[Path]
    metrics: List[Mapping[str, object]]
    images_dir: Path


DEFAULT_VARIANT = ExperimentVariant(
    name="gradient_sweep",
    base_params=[0.19971691],
    perturbation=[1e-2],
    sweep=SweepSettings(center=0.199, width=0.04, num=15, filename_stem="gradient_sweep"),
)


def run_experiment(
    variants: Union[Sequence[ExperimentVariant], ExperimentVariant, None] = None,
    *,
    results_root: Path | str | None = None,
    auto_init_engine: bool = False,
) -> List[ExperimentResult]:
    """Execute one or more variants, writing under ``results/data`` by default."""
    if variants is None:
        selected = [DEFAULT_VARIANT]
    elif isinstance(variants, ExperimentVariant):
        selected = [variants]
    else:
        selected = list(variants)

    if auto_init_engine and getattr(COMSOL_Model, "_engine", None) is None:
        COMSOL_Model.init_engine()

    base_dir = Path.cwd() if results_root is None else Path(results_root)
    results_dir = base_dir if base_dir.name == "results" else base_dir / "results"
    timestamp_dir = results_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    results: List[ExperimentResult] = []
    for variant in selected:
        run_dir = _prepare_run_dir(timestamp_dir, variant.name)
        res = _run_variant(variant, run_dir)
        results.append(res)
    return results


def _run_variant(variant: ExperimentVariant, run_dir: Path) -> ExperimentResult:
    designer = SymmetricTransmonDesign()
    evaluator = AdjointEvaluator(designer)
    evaluator.param_perturbation = np.asarray(variant.perturbation, dtype=float)
    evaluator.fwd_source_strength = float(variant.source_strength)

    sweep = variant.sweep
    source_locations = list(variant.source_locations or [tuple(evaluator.fwd_source_location)])

    outputs: List[Path] = []
    metrics: List[Mapping[str, object]] = []
    images_dir = run_dir / "images"
    for index, location in enumerate(source_locations):
        evaluator.fwd_source_location = list(location)
        optimiser = Optimiser(np.asarray(variant.base_params, dtype=float), 0.01, evaluator)

        if sweep.reuse_fields:
            prefix = f"{variant.name}__src{index:03d}"
            new_outputs = _run_reuse_sweep(optimiser, sweep, run_dir, prefix, resume=variant.resume)
        else:
            name = f"{variant.name}__src{index:03d}"
            new_outputs = _run_standard_sweep(optimiser, sweep, run_dir, name, resume=variant.resume)

        outputs.extend(new_outputs)
        for out in new_outputs:
            summary = _summarise_dat(out)
            summary.update({
                "source_index": index,
                "source_location": tuple(location),
                "variant": variant.name,
            })
            metrics.append(summary)

    if variant.gradient_descent:
        gd_outputs, gd_metrics = _run_gradient_descent(
            evaluator,
            variant.gradient_descent,
            run_dir,
            resume=variant.resume,
        )
        outputs.extend(gd_outputs)
        for entry in gd_metrics:
            entry.update({"phase": "gradient_descent", "variant": variant.name})
            metrics.append(entry)
    metadata_path = run_dir / f"{variant.name}__metadata.json"
    metadata = {
        "variant": variant.name,
        "base_params": list(variant.base_params),
        "perturbation": list(variant.perturbation),
        "source_locations": [list(loc) for loc in source_locations],
        "sweep": dataclasses.asdict(sweep),
        "gradient_descent": dataclasses.asdict(variant.gradient_descent) if variant.gradient_descent else None,
        "metadata": dict(variant.metadata),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    outputs.append(metadata_path)

    return ExperimentResult(
        variant=variant,
        run_dir=run_dir,
        outputs=outputs,
        metrics=metrics,
        images_dir=images_dir,
    )


def _run_standard_sweep(
    optimiser: Optimiser,
    sweep: SweepSettings,
    run_dir: Path,
    name: str,
    *,
    resume: bool,
) -> List[Path]:
    output = run_dir / f"{name}.dat"
    if resume and output.exists():
        return [output]

    filename = str(output.with_suffix(""))
    optimiser.sweep(
        center=sweep.center,
        width=sweep.width,
        num=sweep.num,
        adj_rotation=sweep.adjoint_rotation,
        filename=filename,
    )
    return [output]


def _run_reuse_sweep(
    optimiser: Optimiser,
    sweep: SweepSettings,
    run_dir: Path,
    prefix: str,
    *,
    resume: bool,
) -> List[Path]:
    outputs: List[Path] = []
    for angle in sweep.angles:
        tag = f"{prefix}_ang{angle:.4f}rad"
        dat_path = run_dir / f"{tag}.dat"
        if resume and dat_path.exists():
            outputs.append(dat_path)
    optimiser.sweep_reusing_fields(
        center=sweep.center,
        width=sweep.width,
        num=sweep.num,
        angles=tuple(sweep.angles),
        filename_base=str(run_dir),
    )
    for angle in sweep.angles:
        tag = f"{prefix}__ang{angle:.4f}rad"
        outputs.append(run_dir / f"{tag}.dat")
    return outputs


def _run_gradient_descent(
    evaluator: AdjointEvaluator,
    gd: GradientDescentSettings,
    run_dir: Path,
    *,
    resume: bool,
) -> Tuple[List[Path], List[Mapping[str, object]]]:
    history_path = run_dir / f"{gd.filename_stem}_history.json"

    if resume and history_path.exists():
        payload = json.loads(history_path.read_text())
        return [history_path], [_summarise_gradient_descent(payload, history_path)]

    perturbation = (
        np.asarray(gd.perturbation, dtype=float)
        if gd.perturbation is not None
        else evaluator.param_perturbation
    )
    params = np.asarray(gd.initial_param, dtype=float)
    lr = float(gd.lr)
    params_history = [float(np.asarray(params).ravel()[0])]
    loss_history: List[float] = []
    grad_history: List[Tuple[float, float]] = []

    for _ in range(int(gd.steps)):
        grad, loss = evaluator.evaluate(params, perturbation, verbose=False)
        grad_arr = np.asarray(grad).ravel()[0]
        grad_complex = complex(grad_arr)
        loss_val = float(np.asarray(loss).ravel()[0])

        grad_history.append((grad_complex.real, grad_complex.imag))
        loss_history.append(loss_val)

        params = params - lr * np.array([grad_complex.imag], dtype=float)
        params_history.append(float(np.asarray(params).ravel()[0]))

    payload = {
        "param_history": params_history,
        "loss_history": loss_history,
        "gradient_history": [[g[0], g[1]] for g in grad_history],
    }
    history_path.write_text(json.dumps(payload, indent=2))
    return [history_path], [_summarise_gradient_descent(payload, history_path)]


def _prepare_run_dir(root: Path, variant_name: str) -> Path:
    return root


def _summarise_dat(dat_path: Path) -> Mapping[str, object]:
    if not dat_path.exists():
        return {"path": str(dat_path), "status": "missing"}
    with dat_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
    try:
        data = np.loadtxt(dat_path, skiprows=1, delimiter="\t")
    except Exception as exc:
        return {"path": str(dat_path), "status": f"unreadable: {exc}"}

    if data.ndim == 1:
        data = data[None, :]

    index = {name: idx for idx, name in enumerate(header)}
    summary: Dict[str, object] = {"path": str(dat_path), "status": "ok"}

    def column(name: str) -> Optional[np.ndarray]:
        idx = index.get(name)
        return data[:, idx] if idx is not None else None

    param = column("param")
    loss = column("loss")
    grad_abs = column("abs_grad")
    grad_real = column("real_grad")
    grad_imag = column("imag_grad")
    empirical = column("empirical_gradient")

    if param is not None:
        summary["param_min"] = float(np.min(param))
        summary["param_max"] = float(np.max(param))
    if loss is not None:
        summary["loss_min"] = float(np.min(loss))
        summary["loss_max"] = float(np.max(loss))
    if grad_abs is not None:
        summary["grad_abs_max"] = float(np.max(np.abs(grad_abs)))
        summary["grad_abs_min"] = float(np.min(np.abs(grad_abs)))
    elif grad_real is not None and grad_imag is not None:
        magnitude = np.hypot(grad_real, grad_imag)
        summary["grad_abs_max"] = float(np.max(magnitude))
        summary["grad_abs_min"] = float(np.min(magnitude))
    if empirical is not None and grad_abs is not None:
        diff = grad_abs - empirical
        summary["grad_empirical_rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    return summary


def _summarise_gradient_descent(payload: Mapping[str, object], history_path: Path) -> Mapping[str, object]:
    param_hist = payload.get("param_history", [])
    loss_hist = payload.get("loss_history", [])
    grad_hist = payload.get("gradient_history", [])

    summary: Dict[str, object] = {"path": str(history_path), "status": "ok"}

    if param_hist:
        summary["initial_param"] = float(param_hist[0])
        summary["final_param"] = float(param_hist[-1])
    if loss_hist:
        loss_arr = np.asarray(loss_hist, dtype=float)
        summary["loss_min"] = float(np.min(loss_arr))
        summary["loss_max"] = float(np.max(loss_arr))
        summary["final_loss"] = float(loss_arr[-1])
    if grad_hist:
        grads = [complex(*pair) for pair in grad_hist]
        abs_vals = np.abs(grads)
        summary["grad_abs_max"] = float(np.max(abs_vals))
        summary["final_grad_abs"] = float(abs_vals[-1])
    return summary


__all__ = [
    "ExperimentVariant",
    "ExperimentResult",
    "GradientDescentSettings",
    "SweepSettings",
    "run_experiment",
]
