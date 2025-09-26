"""Plotting utilities for COMSOL adjoint experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

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

def load_dat(path: Path) -> Mapping[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
    data = np.loadtxt(path, skiprows=1, delimiter="\t")
    if data.ndim == 1:
        data = data[None, :]
    return {name: data[:, idx] for idx, name in enumerate(header)}


def gradient_suite(
    data_path: Path,
    *,
    output_root: Path = Path("images"),
    base_name: Optional[str] = None,
    include_empirical: bool = True,
    style: PlotStyle | None = None,
) -> Path:
    columns = load_dat(data_path)
    param = columns["param"]
    grad_real = columns.get("real_grad")
    grad_imag = columns.get("imag_grad")
    grad_abs = columns.get("abs_grad")
    if grad_abs is None and grad_real is not None and grad_imag is not None:
        grad_abs = np.hypot(grad_real, grad_imag)
    loss = columns.get("loss")
    empirical = columns.get("empirical_gradient") if include_empirical else None

    output_root = Path(output_root)
    if output_root.exists() and output_root.is_dir() and output_root.name != "images":
        output_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    name = base_name or data_path.stem

    style = style or PlotStyle()
    with plt.rc_context(style.rc_params):
        fig, axes = plt.subplots(2, 2, sharex="col", figsize=style.figure_size, constrained_layout=True)
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
        axes[1, 1].legend()
        axes[1, 1].set_title("Comparison")

        for ax in axes.flat:
            ax.set_xlabel("parameter")
            ax.set_ylabel("value")

        out_path = output_root / f"{name}_gradient_suite.png"
        fig.savefig(out_path, dpi=style.dpi)
        plt.close(fig)

    return out_path


def gradient_overlay(
    data_path: Path,
    *,
    output_root: Path = Path("images"),
    base_name: Optional[str] = None,
    style: PlotStyle | None = None,
) -> Path:
    columns = load_dat(data_path)
    param = columns["param"]
    grad_abs = columns.get("abs_grad")
    if grad_abs is None:
        grad_abs = np.hypot(columns["real_grad"], columns["imag_grad"])
    empirical = columns.get("empirical_gradient")

    output_root = Path(output_root)
    if output_root.exists() and output_root.is_dir() and output_root.name != "images":
        output_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    name = base_name or data_path.stem

    style = style or PlotStyle()
    with plt.rc_context(style.rc_params):
        fig, ax = plt.subplots(figsize=style.figure_size)
        ax.plot(param, grad_abs, label="adjoint", color="C0")
        if empirical is not None:
            ax.plot(param, empirical, label="empirical", color="C3", linestyle="--")
        ax.set_xlabel("parameter")
        ax.set_ylabel("|dg/dp|")
        ax.set_title("Adjoint vs empirical gradient")
        ax.legend()

        out_path = output_root / f"{name}_gradient_overlay.png"
        fig.savefig(out_path, dpi=style.dpi)
        plt.close(fig)

    return out_path


def geometry_family(
    parameters: Sequence[Sequence[float]],
    *,
    output_root: Path = Path("images"),
    base_name: str = "geometry_family",
    labels: Optional[Sequence[str]] = None,
    style: PlotStyle | None = None,
) -> Path:
    designer = SymmetricTransmonDesign()
    output_root = Path(output_root)
    if output_root.exists() and output_root.is_dir() and output_root.name != "images":
        output_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)

    style = style or PlotStyle()
    with plt.rc_context(style.rc_params):
        fig, ax = plt.subplots(figsize=style.figure_size)
        for idx, params in enumerate(parameters):
            geom = designer.geometry(np.asarray(params, dtype=float))
            _plot_polygons(ax, geom, label=labels[idx] if labels else str(params), color=f"C{idx % 10}")
        ax.set_aspect("equal")
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
        ax.set_title("Symmetric transmon family")
        if labels:
            ax.legend(loc="upper right")

        out_path = output_root / f"{base_name}_geometry.png"
        fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


def boundary_velocity(
    params: Sequence[float],
    perturbation: Sequence[float],
    *,
    output_root: Path = Path("images"),
    base_name: str = "boundary_velocity",
    max_arrows: int = 6,
    style: PlotStyle | None = None,
) -> Path:
    designer = SymmetricTransmonDesign()
    velocities, refs, near = designer.compute_boundary_velocity(np.asarray(params, dtype=float), np.asarray(perturbation, dtype=float))

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

    geom_ref = designer.geometry(np.asarray(params, dtype=float))
    geom_pert = designer.geometry(np.asarray(params, dtype=float) + np.asarray(perturbation, dtype=float))

    output_root = Path(output_root)
    if output_root.exists() and output_root.is_dir() and output_root.name != "images":
        output_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)

    style = style or PlotStyle()
    with plt.rc_context(style.rc_params):
        fig, ax = plt.subplots(figsize=style.figure_size)
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

        out_path = output_root / f"{base_name}_boundary_velocity.png"
        fig.savefig(out_path, dpi=style.dpi, bbox_inches="tight")
        plt.close(fig)

    return out_path


__all__ = [
    "PlotStyle",
    "load_dat",
    "gradient_suite",
    "gradient_overlay",
    "geometry_family",
    "boundary_velocity",
]
