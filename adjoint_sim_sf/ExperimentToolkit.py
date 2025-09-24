"""Automation toolkit for COMSOL adjoint experiments.

Derived directly from the historical notebooks, this module keeps the experiment surface
minimal: define one or more `ExperimentVariant` instances and pass them to `run_experiment`.
Each run produces deterministic TSV artefacts plus JSON metadata, enabling reproducible sweeps
without relying on notebooks.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
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
    conjugations: Sequence[bool] = (False,)
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
class RunResult:
    variant: ExperimentVariant
    run_dir: Path
    outputs: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: List[Mapping[str, object]] = field(default_factory=list)

    def as_dict(self) -> Mapping[str, object]:
        return {
            "variant": dataclasses.asdict(self.variant),
            "run_dir": str(self.run_dir),
            "outputs": [str(p) for p in self.outputs],
            "errors": self.errors,
            "metrics": self.metrics,
        }


DEFAULT_VARIANT = ExperimentVariant(
    name="gradient_sweep",
    base_params=[0.19971691],
    perturbation=[1e-2],
    sweep=SweepSettings(center=0.199, width=0.04, num=15, filename_stem="gradient_sweep"),
    metadata={"origin": "quickcheck.ipynb"},
)


class ExperimentToolkit:
    """High-level orchestrator for COMSOL adjoint experiments."""

    def __init__(
        self,
        output_root: Path | str = Path("saved_data") / "automation_runs",
        *,
        auto_init_engine: bool = False,
        log_level: int = logging.INFO,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("experiment_toolkit")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)s | %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

        if auto_init_engine and getattr(COMSOL_Model, "_engine", None) is None:
            self.logger.info("Initialising COMSOL engine")
            COMSOL_Model.init_engine()

    def run(self, variants: Sequence[ExperimentVariant], *, stop_on_error: bool = False) -> List[RunResult]:
        results: List[RunResult] = []
        for variant in variants:
            result = self._run_variant(variant, stop_on_error=stop_on_error)
            results.append(result)
        summary_path = self.output_root / f"run_summary_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        summary_path.write_text(json.dumps([r.as_dict() for r in results], indent=2))
        self.logger.info("Saved run summary to %s", summary_path)
        return results

    def _run_variant(self, variant: ExperimentVariant, *, stop_on_error: bool) -> RunResult:
        run_dir = self._prepare_run_dir(variant.name)
        result = RunResult(variant=variant, run_dir=run_dir)
        log_path = run_dir / "run.log"
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.info("Starting variant %s", variant.name)
        try:
            self._execute_variant(variant, run_dir, result)
        except Exception as exc:
            message = f"Variant {variant.name} failed: {exc}"
            self.logger.exception(message)
            result.errors.append(f"{message}\n{traceback.format_exc()}")
            if stop_on_error:
                raise
        finally:
            self.logger.removeHandler(file_handler)
            file_handler.close()
        return result

    def _execute_variant(self, variant: ExperimentVariant, run_dir: Path, result: RunResult) -> None:
        designer = SymmetricTransmonDesign()
        evaluator = AdjointEvaluator(designer)
        evaluator.param_perturbation = np.asarray(variant.perturbation, dtype=float)
        evaluator.fwd_source_strength = float(variant.source_strength)

        sweep = variant.sweep
        source_locations = list(variant.source_locations or [tuple(evaluator.fwd_source_location)])

        for idx, location in enumerate(source_locations):
            evaluator.fwd_source_location = list(location)
            self.logger.info("Running sweep %s | source #%d -> %s", variant.name, idx, location)
            sweep_dir = run_dir / f"sweep_{idx:03d}"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            optimiser = Optimiser(np.asarray(variant.base_params, dtype=float), 0.01, evaluator)

            if sweep.reuse_fields:
                outputs = self._run_reuse_sweep(optimiser, sweep, sweep_dir, resume=variant.resume)
            else:
                outputs = self._run_standard_sweep(optimiser, sweep, sweep_dir, resume=variant.resume)

            result.outputs.extend(outputs)
            for out in outputs:
                metrics = self._summarise_dat(out)
                metrics.update({
                    "source_index": idx,
                    "source_location": tuple(location),
                    "variant": variant.name,
                })
                result.metrics.append(metrics)

        if variant.gradient_descent:
            gd_outputs, gd_metrics = self._run_gradient_descent(
                evaluator,
                variant.gradient_descent,
                run_dir,
                resume=variant.resume,
            )
            result.outputs.extend(gd_outputs)
            for metrics in gd_metrics:
                metrics.update({
                    "phase": "gradient_descent",
                    "variant": variant.name,
                })
                result.metrics.append(metrics)

        metadata_path = run_dir / "metadata.json"
        metadata = {
            "variant": variant.name,
            "base_params": list(variant.base_params),
            "perturbation": list(variant.perturbation),
            "source_locations": [list(loc) for loc in source_locations],
            "sweep": dataclasses.asdict(sweep),
            "gradient_descent": dataclasses.asdict(variant.gradient_descent) if variant.gradient_descent else None,
            "notes": dict(variant.metadata),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        result.outputs.append(metadata_path)
        self.logger.info("Finished variant %s", variant.name)

    def _run_standard_sweep(
        self,
        optimiser: Optimiser,
        sweep: SweepSettings,
        sweep_dir: Path,
        *,
        resume: bool,
    ) -> List[Path]:
        output = sweep_dir / f"{sweep.filename_stem}.dat"
        if resume and output.exists():
            self.logger.info("Skipping existing sweep output %s", output)
            return [output]

        filename = str(output.with_suffix(""))
        optimiser.sweep(
            center=sweep.center,
            width=sweep.width,
            num=sweep.num,
            adj_rotation=sweep.adjoint_rotation,
            conjugations=tuple(sweep.conjugations),
            filename=filename,
        )
        if not output.exists():
            output = output.with_suffix(".dat")
        return [output]

    def _run_reuse_sweep(
        self,
        optimiser: Optimiser,
        sweep: SweepSettings,
        sweep_dir: Path,
        *,
        resume: bool,
    ) -> List[Path]:
        outputs: List[Path] = []
        base = sweep_dir / sweep.filename_stem
        for conj in sweep.conjugations:
            for angle in sweep.angles:
                tag = f"conj={'T' if conj else 'F'}_ang={angle:.4f}rad"
                stem = base / tag
                stem.parent.mkdir(parents=True, exist_ok=True)
                dat_path = stem.with_suffix(".dat")
                if resume and dat_path.exists():
                    self.logger.info("Skipping existing field-reuse output %s", dat_path)
                    outputs.append(dat_path)
                    continue
        optimiser.sweep_reusing_fields(
            center=sweep.center,
            width=sweep.width,
            num=sweep.num,
            angles=tuple(sweep.angles),
            conjugations=tuple(sweep.conjugations),
            filename_base=str(base),
        )
        for conj in sweep.conjugations:
            for angle in sweep.angles:
                tag = f"conj={'T' if conj else 'F'}_ang={angle:.4f}rad"
                dat_path = (base / tag).with_suffix(".dat")
                outputs.append(dat_path)
        return outputs

    def _run_gradient_descent(
        self,
        evaluator: AdjointEvaluator,
        gd: GradientDescentSettings,
        run_dir: Path,
        *,
        resume: bool,
    ) -> Tuple[List[Path], List[Mapping[str, object]]]:
        out_dir = run_dir / gd.filename_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        history_path = out_dir / "history.json"

        if resume and history_path.exists():
            payload = json.loads(history_path.read_text())
            metrics = self._summarise_gradient_descent(payload, history_path)
            return [history_path], [metrics]

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
        metrics = self._summarise_gradient_descent(payload, history_path)
        return [history_path], [metrics]

    def _prepare_run_dir(self, variant_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.output_root / variant_name / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _summarise_dat(self, dat_path: Path) -> Mapping[str, object]:
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

    def _summarise_gradient_descent(
        self,
        payload: Mapping[str, object],
        history_path: Path,
    ) -> Mapping[str, object]:
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
            grads = [complex(real, imag) for real, imag in grad_hist]
            abs_vals = np.abs(grads)
            summary["grad_abs_max"] = float(np.max(abs_vals))
            summary["final_grad_abs"] = float(abs_vals[-1])
        return summary


def _preset_variants() -> Mapping[str, ExperimentVariant]:
    angles: List[float] = []
    for q in range(2, 16):
        for k in range(0, 2 * q):
            angles.append(np.pi * k / q)
    unique_angles = sorted(set(angles))

    return {
        "gradient_sweep": DEFAULT_VARIANT,
        "rotation_sweep": ExperimentVariant(
            name="rotation_sweep",
            base_params=[0.199],
            perturbation=[1e-2],
            sweep=SweepSettings(
                center=0.199,
                width=0.035,
                num=200,
                reuse_fields=True,
                angles=tuple(unique_angles),
                conjugations=(False,),
                filename_stem="rotation_sweep",
            ),
            metadata={"origin": "check_adjoint_rotation.ipynb"},
        ),
        "source_grid": ExperimentVariant(
            name="source_grid",
            base_params=[0.199],
            perturbation=[0.01],
            sweep=SweepSettings(center=0.199, width=0.035, num=1, filename_stem="source_grid"),
            source_locations=[
                [float(x), float(y), 100e-6]
                for x in np.linspace(-550e-6, +550e-6, 6)
                for y in np.linspace(-550e-6, +550e-6, 6)
            ],
            metadata={"origin": "check_source_location.ipynb"},
        ),
        "gradient_descent": ExperimentVariant(
            name="gradient_descent",
            base_params=[0.210],
            perturbation=[0.01],
            sweep=SweepSettings(center=0.210, width=0.0, num=1, filename_stem="placeholder"),
            gradient_descent=GradientDescentSettings(initial_param=[0.210], steps=8, lr=0.01),
            metadata={"origin": "gradient_descent.ipynb"},
        ),
    }


def load_variants_from_json(config_path: Path) -> List[ExperimentVariant]:
    raw = json.loads(config_path.read_text())
    entries = raw.get("variants", raw)
    variants: List[ExperimentVariant] = []
    for entry in entries:
        sweep = entry.get("sweep", {})
        gd = entry.get("gradient_descent")
        variant = ExperimentVariant(
            name=entry["name"],
            base_params=entry.get("base_params", [0.199]),
            perturbation=entry.get("perturbation", [1e-5]),
            sweep=SweepSettings(**sweep),
            source_locations=entry.get("source_locations"),
            source_strength=entry.get("source_strength", 1e-2),
            metadata=entry.get("metadata", {}),
            gradient_descent=GradientDescentSettings(**gd) if gd else None,
            resume=entry.get("resume", True),
        )
        variants.append(variant)
    return variants


def run_experiment(
    variants: Union[Sequence[ExperimentVariant], ExperimentVariant, None] = None,
    *,
    output_root: Path | str = Path("saved_data") / "automation_runs",
    stop_on_error: bool = False,
    auto_init_engine: bool = False,
    log_level: int = logging.INFO,
) -> List[RunResult]:
    if variants is None:
        selected = [DEFAULT_VARIANT]
    elif isinstance(variants, ExperimentVariant):
        selected = [variants]
    else:
        selected = list(variants)
    toolkit = ExperimentToolkit(
        output_root=output_root,
        auto_init_engine=auto_init_engine,
        log_level=log_level,
    )
    return toolkit.run(selected, stop_on_error=stop_on_error)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run automated COMSOL adjoint experiments")
    parser.add_argument("command", choices=["list", "run"], help="List presets or execute variants")
    parser.add_argument("--variant", action="append", dest="variants", help="Preset variant name (repeatable)")
    parser.add_argument("--config", type=Path, help="Path to JSON configuration file")
    parser.add_argument("--output-root", type=Path, default=Path("saved_data") / "automation_runs")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--auto-init-engine", action="store_true", help="Initialise COMSOL engine if needed")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    presets = _preset_variants()
    if args.command == "list":
        for name, variant in presets.items():
            print(f"{name}: {variant.metadata.get('origin', 'preset')}")
        if args.config:
            for variant in load_variants_from_json(args.config):
                print(f"{variant.name} (config)")
        return 0

    selected: List[ExperimentVariant] = []
    if args.variants:
        for name in args.variants:
            if name not in presets:
                parser.error(f"Unknown preset variant '{name}'")
            selected.append(presets[name])
    if args.config:
        selected.extend(load_variants_from_json(args.config))

    if not selected:
        selected = [DEFAULT_VARIANT]

    if args.no_resume:
        selected = [dataclasses.replace(v, resume=False) for v in selected]

    run_experiment(
        selected,
        output_root=args.output_root,
        stop_on_error=args.stop_on_error,
        auto_init_engine=args.auto_init_engine,
        log_level=getattr(logging, args.log_level.upper()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
