"""Reusable plotting pipeline for COMSOL adjoint experiments.

The module recreates the visuals from the notebooks while enforcing consistent styling and
repeatable filenames. Plot functions live in a registry so new visuals can be added without
changing the core orchestration.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from adjoint_sim_sf.ParametricDesign import SymmetricTransmonDesign


@dataclass(frozen=True)
class PlotStyle:
    figure_size: tuple[float, float] = (6.0, 4.5)
    dpi: int = 300
    rc_params: Mapping[str, object] = field(
        default_factory=lambda: {
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "lines.linewidth": 1.6,
        }
    )

    @contextlib.contextmanager
    def use(self):
        previous = plt.rcParams.copy()
        try:
            plt.rcParams.update(self.rc_params)
            yield
        finally:
            plt.rcParams.update(previous)


class PlotRegistry:
    """Simple registry for plot functions."""

    def __init__(self) -> None:
        self._plots: Dict[str, Callable[["VizPipeline", MutableMapping[str, object]], List[Path]]] = {}

    def register(
        self, name: str
    ) -> Callable[[Callable[["VizPipeline", MutableMapping[str, object]], List[Path]]], Callable[["VizPipeline", MutableMapping[str, object]], List[Path]]]:
        def decorator(func: Callable[["VizPipeline", MutableMapping[str, object]], List[Path]]):
            self._plots[name] = func
            return func

        return decorator

    def get(self, name: str) -> Callable[["VizPipeline", MutableMapping[str, object]], List[Path]]:
        if name not in self._plots:
            available = ", ".join(sorted(self._plots))
            raise KeyError(f"Unknown plot '{name}'. Available: {available}")
        return self._plots[name]

    def list(self) -> Sequence[str]:
        return tuple(sorted(self._plots))


plot_registry = PlotRegistry()


@dataclass
class VizPipeline:
    """Coordinator for registered plotting routines."""

    output_root: Path = Path("Images") / "automation"
    style: PlotStyle = field(default_factory=PlotStyle)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def plot(self, name: str, **kwargs) -> List[Path]:
        func = plot_registry.get(name)
        payload: MutableMapping[str, object] = dict(kwargs)
        payload.setdefault("base_name", name)
        return func(self, payload)

    def available(self) -> Sequence[str]:
        return plot_registry.list()

    def verify(self, paths: Iterable[Path]) -> Mapping[str, bool]:
        return {str(path): path.exists() for path in paths}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dat_table(path: Path) -> Mapping[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
    data = np.loadtxt(path, skiprows=1, delimiter="\t")
    if data.ndim == 1:
        data = data[None, :]
    return {name: data[:, idx] for idx, name in enumerate(header)}


def _ensure_abs_grad(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    if "abs_grad" in columns:
        return columns["abs_grad"]
    if "real_grad" in columns and "imag_grad" in columns:
        return np.hypot(columns["real_grad"], columns["imag_grad"])
    raise KeyError("No gradient magnitude columns present")


def _plot_polygons(ax: plt.Axes, geom: MultiPolygon | Polygon, *, label: Optional[str] = None, color: Optional[str] = None) -> None:
    geoms = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    for idx, poly in enumerate(geoms):
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color=color or f"C{idx % 10}", linewidth=1.8, label=label if idx == 0 else None)
        for ring in poly.interiors:
            xs, ys = ring.xy
            ax.plot(xs, ys, color=color or f"C{idx % 10}", linewidth=1.2, linestyle="--")


# ---------------------------------------------------------------------------
# Registered plots
# ---------------------------------------------------------------------------


@plot_registry.register("gradient_suite")
def _plot_gradient_suite(pipeline: VizPipeline, payload: MutableMapping[str, object]) -> List[Path]:
    data_path = Path(payload["data_path"])  # type: ignore[arg-type]
    base_name = str(payload.get("base_name", data_path.stem))
    include_empirical = bool(payload.get("include_empirical", True))

    columns = _load_dat_table(data_path)
    param = columns.get("param")
    if param is None:
        raise KeyError("Column 'param' missing from data table")
    grad_real = columns.get("real_grad")
    grad_imag = columns.get("imag_grad")
    grad_abs = _ensure_abs_grad(columns)
    loss = columns.get("loss")
    empirical = columns.get("empirical_gradient") if include_empirical else None

    with pipeline.style.use():
        fig, axes = plt.subplots(2, 2, sharex="col", figsize=pipeline.style.figure_size, constrained_layout=True)
        axes[0, 0].plot(param, grad_real, color="C0")
        axes[0, 0].set_title("Adjoint grad (real)")
        axes[0, 1].plot(param, grad_imag, color="C1")
        axes[0, 1].set_title("Adjoint grad (imag)")
        axes[1, 0].plot(param, grad_abs, color="C2")
        axes[1, 0].set_title("Adjoint |grad|")

        axes[1, 1].plot(param, grad_abs, label="adjoint", color="C2")
        if empirical is not None:
            axes[1, 1].plot(param, empirical, label="empirical", color="C3", linestyle="--")
        if loss is not None:
            scale = np.max(np.abs(loss))
            loss_scaled = loss / scale if scale else loss
            axes[1, 1].plot(param, loss_scaled, label="loss (scaled)", color="C4")
        axes[1, 1].set_title("Comparison")
        axes[1, 1].legend()

        for ax in axes.flat:
            ax.set_xlabel("parameter")
            ax.set_ylabel("value")

        out_path = pipeline.output_root / f"{base_name}_gradient_suite.png"
        fig.savefig(out_path, dpi=pipeline.style.dpi)
        plt.close(fig)

    return [out_path]


@plot_registry.register("gradient_overlay")
def _plot_gradient_overlay(pipeline: VizPipeline, payload: MutableMapping[str, object]) -> List[Path]:
    data_path = Path(payload["data_path"])  # type: ignore[arg-type]
    base_name = str(payload.get("base_name", data_path.stem))

    columns = _load_dat_table(data_path)
    param = columns.get("param")
    grad_abs = _ensure_abs_grad(columns)
    empirical = columns.get("empirical_gradient")

    with pipeline.style.use():
        fig, ax = plt.subplots(figsize=pipeline.style.figure_size)
        ax.plot(param, grad_abs, label="adjoint", color="C0")
        if empirical is not None:
            ax.plot(param, empirical, label="empirical", color="C3", linestyle="--")
        ax.set_xlabel("parameter")
        ax.set_ylabel("|dg/dp|")
        ax.set_title("Adjoint vs empirical gradient")
        ax.legend()

        out_path = pipeline.output_root / f"{base_name}_gradient_overlay.png"
        fig.savefig(out_path, dpi=pipeline.style.dpi)
        plt.close(fig)
    return [out_path]


@plot_registry.register("geometry_family")
def _plot_geometry_family(pipeline: VizPipeline, payload: MutableMapping[str, object]) -> List[Path]:
    params = payload.get("parameters")
    if params is None:
        raise ValueError("'parameters' is required for geometry_family plot")
    parameter_sets = [np.atleast_1d(p).astype(float) for p in params]
    labels = payload.get("labels")
    base_name = str(payload.get("base_name", "geometry_family"))

    designer = SymmetricTransmonDesign()

    with pipeline.style.use():
        fig, ax = plt.subplots(figsize=pipeline.style.figure_size)
        for idx, (theta, label) in enumerate(zip(parameter_sets, labels or parameter_sets)):
            geom = designer.geometry(theta)
            _plot_polygons(ax, geom, label=str(label), color=f"C{idx % 10}")
        ax.set_aspect("equal")
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        ax.set_title("Symmetric transmon family")
        if labels:
            ax.legend(loc="upper right")

        out_path = pipeline.output_root / f"{base_name}_geometry.png"
        fig.savefig(out_path, dpi=pipeline.style.dpi, bbox_inches="tight")
        plt.close(fig)
    return [out_path]


@plot_registry.register("boundary_velocity")
def _plot_boundary_velocity(pipeline: VizPipeline, payload: MutableMapping[str, object]) -> List[Path]:
    params = np.atleast_1d(payload.get("params"))
    perturbation = np.atleast_1d(payload.get("perturbation"))
    base_name = str(payload.get("base_name", "boundary_velocity"))
    max_arrows = int(payload.get("max_arrows", 6))

    designer = SymmetricTransmonDesign()
    velocities, refs, near = designer.compute_boundary_velocity(params, perturbation)

    v_flat: List[float] = []
    p_flat: List[tuple[float, float]] = []
    q_flat: List[tuple[float, float]] = []
    for v_poly, r_poly, q_poly in zip(velocities, refs, near):
        for v_ring, r_ring, q_ring in zip(v_poly, r_poly, q_poly):
            for v, r, q in zip(v_ring, r_ring, q_ring):
                v_flat.append(float(v))
                p_flat.append((float(r[0]), float(r[1])))
                q_flat.append((float(q[0]), float(q[1])))

    if not v_flat:
        raise RuntimeError("No boundary velocities computed")

    indices = np.linspace(0, len(v_flat) - 1, min(max_arrows, len(v_flat))).astype(int)
    sample = [(v_flat[i], p_flat[i], q_flat[i]) for i in indices]

    geom_ref = designer.geometry(params)
    geom_pert = designer.geometry(params + perturbation)

    with pipeline.style.use():
        fig, ax = plt.subplots(figsize=pipeline.style.figure_size)
        _plot_polygons(ax, geom_ref, label="reference", color="C0")
        _plot_polygons(ax, geom_pert, label="perturbed", color="C1")

        for idx, (vel, ref_pt, tgt_pt) in enumerate(sample, start=1):
            ax.annotate(
                "",
                xy=tgt_pt,
                xytext=ref_pt,
                arrowprops=dict(arrowstyle="->", linewidth=1.5, color="C2"),
            )
            ax.text(
                tgt_pt[0],
                tgt_pt[1],
                f"{idx}:{vel:.3f}",
                color="C2",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

        ax.set_aspect("equal")
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        ax.set_title("Boundary velocity sample")
        ax.legend(loc="upper right")

        out_path = pipeline.output_root / f"{base_name}_boundary_velocity.png"
        fig.savefig(out_path, dpi=pipeline.style.dpi, bbox_inches="tight")
        plt.close(fig)
    return [out_path]
