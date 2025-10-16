# ExperimentViz.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Plotter:
    @staticmethod
    def gradient_contour_2d(
        all_params,
        loss_array,
        numeric_positions,
        numeric_gradients,
        adj_positions,
        adj_gradients,
        title=None,
        skip_idx_numeric=None,
        skip_idx_adjoint=None,
        quiver_scale=None,
        quiver_width=0.003,
    ):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Interpolate loss for contours
        xi = np.linspace(all_params[:, 0].min(), all_params[:, 0].max(), 100)
        yi = np.linspace(all_params[:, 1].min(), all_params[:, 1].max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata(all_params, loss_array, (Xi, Yi), method="cubic")

        # Panel 1: Numerical gradients
        ax1 = axes[0]
        contour1 = ax1.contourf(Xi, Yi, Zi, levels=20, cmap="viridis")
        plt.colorbar(contour1, ax=ax1, label="Loss")

        if numeric_positions.size and numeric_gradients.size:
            if skip_idx_numeric is None:
                idx = np.arange(len(numeric_positions), dtype=int)
            else:
                idx = skip_idx_numeric
            ax1.quiver(
                numeric_positions[idx, 0],
                numeric_positions[idx, 1],
                -numeric_gradients[idx, 0],
                -numeric_gradients[idx, 1],
                alpha=0.6,
                scale=quiver_scale,
                scale_units='xy',  # Add this
                angles='xy',       # Add this
                width=quiver_width,
                headwidth=3,       # Add this - controls arrowhead width
                headlength=5,      # Add this - controls arrowhead length
                headaxislength=4.5, # Add this - controls arrowhead shape
                label="Numeric grad",
            )

        ax1.set_title("Loss Contour with Numerical Gradients")
        ax1.set_xlabel("Param 1")
        ax1.set_ylabel("Param 2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Adjoint gradients
        ax2 = axes[1]
        contour2 = ax2.contourf(Xi, Yi, Zi, levels=20, cmap="viridis")
        plt.colorbar(contour2, ax=ax2, label="Loss")

        if adj_positions.size and adj_gradients.size:
            if skip_idx_adjoint is None:
                idx = np.arange(len(adj_positions), dtype=int)
            else:
                idx = skip_idx_adjoint
            ax2.quiver(
                adj_positions[idx, 0],
                adj_positions[idx, 1],
                -adj_gradients[idx, 0],
                -adj_gradients[idx, 1],
                alpha=0.6,
                scale=quiver_scale,
                scale_units='xy',     
                angles='xy',           
                width=quiver_width,
                headwidth=3,           
                headlength=5,          
                headaxislength=4.5,    
                label="Adjoint grad",
            )

        ax2.set_title("Loss Contour with Adjoint Gradients")
        ax2.set_xlabel("Param 1")
        ax2.set_ylabel("Param 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=1.03)

        plt.tight_layout()
        return fig, axes

    @staticmethod
    def gradient_mag_comparison(numeric_gradients, adjoint_gradients, scale=1.0, uselog=True, title=None):
        num_mag = np.linalg.norm(numeric_gradients, axis=1) if numeric_gradients.size else np.array([])
        adj_mag = np.linalg.norm(adjoint_gradients, axis=1) if adjoint_gradients.size else np.array([])

        fig, ax = plt.subplots(figsize=(10, 8))

        if num_mag.size and adj_mag.size:
            ax.scatter(num_mag, adj_mag * scale, alpha=0.5, s=30)

            max_val = max(num_mag.max(), (adj_mag * scale).max())
            min_val = min(num_mag.min(), (adj_mag * scale).min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement', alpha=0.7)

        ax.set_xlabel("Numerical Gradient Magnitude", fontsize=12)
        ax.set_ylabel("Adjoint Gradient Magnitude", fontsize=12)
        ax.set_title(title or "Gradient Magnitude Comparison", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        if uselog:
            ax.set_xscale("log")
            ax.set_yscale("log")

        return fig, ax
