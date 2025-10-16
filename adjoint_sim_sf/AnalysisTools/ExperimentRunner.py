# ExperimentRunner.py
import json
import os
from datetime import datetime
import numpy as np

from .ExperimentViz import Plotter  # assumes same folder; adjust import path if needed

class Experiment:
    def __init__(self, name=None, verbose=True):
        self.name = name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
        self.dir = f"experiments/{self.name}"
        os.makedirs(self.dir, exist_ok=True)
        if verbose:
            print(f"Commencing experiment: {self.dir}")
        self.verbose = verbose

    def path(self, filename):
        return os.path.join(self.dir, filename)

    def save_results(self, results, filename):
        with open(self.path(filename), "w") as f:
            for result in results:
                f.write(json.dumps(result, default=self._json_convert) + "\n")

    def save_config(self, config, filename="config.json"):
        with open(self.path(filename), "w") as f:
            json.dump(config, f, indent=2, default=self._json_convert)

    @staticmethod
    def _json_convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} not JSON serializable")

    # ---------- Data IO ----------

    def load_jsonl(self, rel_path, loss_key="loss", grad_key="grad"):
        with open(self.path(rel_path), "r") as f:
            records = [json.loads(line) for line in f if line.strip()]

        params_array = np.array([r["params"] for r in records], dtype=float)
        loss_array = np.array([r.get(loss_key, np.nan) for r in records], dtype=float)

        # If some rows lack the grad key, fill with NaNs of correct dim
        param_dim = params_array.shape[1]
        adj_grad_array = np.array(
            [r.get(grad_key, [np.nan] * param_dim) for r in records],
            dtype=float
        )

        return {
            "records": records,
            "params_array": params_array,
            "loss_array": loss_array,
            "adj_grad_array": adj_grad_array,
            "param_dim": param_dim,
            "loss_key": loss_key,
            "grad_key": grad_key,
        }

    # ---------- Numeric gradients (forward-diff on grid) ----------

    def compute_numeric_grads(self, params_array, loss_array, adj_grad_array):
        x_vals = np.unique(params_array[:, 0])
        y_vals = np.unique(params_array[:, 1]) if params_array.shape[1] > 1 else np.array([])

        data_dict = {}
        def key(x, y, prec=12):
            return (round(float(x), prec), round(float(y), prec))

        if params_array.shape[1] >= 2:
            data_dict = {
                key(px, py): (lv, i)
                for i, ((px, py), lv) in enumerate(zip(params_array, loss_array))
            }

        numeric_grad_list = []
        valid_indices = []

        if params_array.shape[1] >= 2 and len(x_vals) and len(y_vals):
            for i, (x, y) in enumerate(params_array):
                x_nexts = x_vals[x_vals > x]
                y_nexts = y_vals[y_vals > y]
                if x_nexts.size == 0 or y_nexts.size == 0:
                    continue
                x_next = x_nexts[0]
                y_next = y_nexts[0]

                k_c = key(x, y)
                k_x = key(x_next, y)
                k_y = key(x, y_next)

                if k_c in data_dict and k_x in data_dict and k_y in data_dict:
                    loss_c = data_dict[k_c][0]
                    loss_x = data_dict[k_x][0]
                    loss_y = data_dict[k_y][0]

                    dx = x_next - x
                    dy = y_next - y

                    grad_x = (loss_x - loss_c) / dx
                    grad_y = (loss_y - loss_c) / dy

                    numeric_grad_list.append([grad_x, grad_y])
                    valid_indices.append(i)

        numeric_grad_array = np.array(numeric_grad_list)
        valid_indices = np.array(valid_indices, dtype=int)

        valid_params = params_array[valid_indices] if valid_indices.size else np.empty((0, params_array.shape[1]))
        valid_loss = loss_array[valid_indices] if valid_indices.size else np.empty((0,))
        valid_adj_grad_array = adj_grad_array[valid_indices] if valid_indices.size else np.empty((0, adj_grad_array.shape[1]))

        if self.verbose:
            print(f"Computed {numeric_grad_array.shape[0]} numerical gradients")

        return {
            "numeric_grad_array": numeric_grad_array,
            "valid_indices": valid_indices,
            "valid_params": valid_params,
            "valid_loss": valid_loss,
            "valid_adj_grad_array": valid_adj_grad_array,
            "adj_grad_array_full": adj_grad_array,
        }

    # ---------- Plotting ----------

    def _compute_skip_idx(self, total, max_to_show):
        if max_to_show <= 0 or max_to_show >= total:
            return np.arange(total, dtype=int)
        step = max(1, total // max_to_show)
        return np.arange(0, total, step, dtype=int)[:max_to_show]

    def plot_contours_if_2d(self, data_dict, grads_dict, max_to_show=0, title_suffix=None, save_name=None):
        params_array = data_dict["params_array"]
        loss_array = data_dict["loss_array"]
        param_dim = data_dict["param_dim"]

        if param_dim != 2:
            if self.verbose:
                print("Param dim != 2: skipping contour/quiver figure.")
            return None, None

        # Positions and skips
        valid_params = grads_dict["valid_params"]
        numeric_grad_array = grads_dict["numeric_grad_array"]

        adj_positions = params_array
        adj_gradients = grads_dict["adj_grad_array_full"]

        idx_num = self._compute_skip_idx(len(valid_params), max_to_show) if len(valid_params) else np.array([], dtype=int)
        idx_adj = self._compute_skip_idx(len(adj_positions), max_to_show) if len(adj_positions) else np.array([], dtype=int)

        title = "Loss Contours with Gradients"
        if title_suffix:
            title += f" – {title_suffix}"

        fig, axes = Plotter.gradient_contour_2d(
            all_params=params_array,
            loss_array=loss_array,
            numeric_positions=valid_params,
            numeric_gradients=numeric_grad_array,
            adj_positions=adj_positions,
            adj_gradients=adj_gradients,
            title=title,
            skip_idx_numeric=idx_num,
            skip_idx_adjoint=idx_adj,
        )

        out_name = save_name or "contours_numeric_vs_adjoint.png"
        fig.savefig(self.path(out_name), dpi=150, bbox_inches="tight")
        if self.verbose:
            print(f"Saved: {self.path(out_name)}")
        return fig, axes

    def plot_magnitude_comparison(self, grads_dict, scale=0.02, uselog=True, title_suffix=None, save_name=None):
        numeric_grad_array = grads_dict["numeric_grad_array"]
        valid_adj_grad_array = grads_dict["valid_adj_grad_array"]

        title = "Gradient Magnitude Comparison (numeric vs adjoint)"
        if title_suffix:
            title += f" – {title_suffix}"

        fig, ax = Plotter.gradient_mag_comparison(
            numeric_gradients=numeric_grad_array,
            adjoint_gradients=valid_adj_grad_array,
            scale=scale,
            uselog=uselog,
            title=title,
        )

        # Summary stats (unscaled, matching your notebook)
        num_mag = np.linalg.norm(numeric_grad_array, axis=1) if numeric_grad_array.size else np.array([])
        adj_mag = np.linalg.norm(valid_adj_grad_array, axis=1) if valid_adj_grad_array.size else np.array([])

        if num_mag.size and adj_mag.size:
            print("\n" + "="*70)
            print("SUMMARY STATISTICS")
            print("="*70)
            corr = np.corrcoef(num_mag, adj_mag)[0, 1]
            print(f"\nCorrelation between gradient magnitudes: {corr:.4f}")
            ratio = np.divide(adj_mag, num_mag, out=np.full_like(adj_mag, np.nan), where=(num_mag != 0))
            print(f"\nRatio statistics (Adjoint/Numeric):")
            print(f"  Mean: {np.nanmean(ratio):.4f}")
            print(f"  Median: {np.nanmedian(ratio):.4f}")
            print(f"  Std: {np.nanstd(ratio):.4f}")
            print("="*70)
        else:
            print("Insufficient data to compute summary statistics.")

        out_name = save_name or "gradient_magnitude_comparison.png"
        fig.savefig(self.path(out_name), dpi=150, bbox_inches="tight")
        if self.verbose:
            print(f"Saved: {self.path(out_name)}")
        return fig, ax

    # ---------- Augmentation ----------

    def augment_results_jsonl(self, in_filename, out_filename="augmented.jsonl", loss_key="loss", grad_key="grad"):
        data = self.load_jsonl(in_filename, loss_key=loss_key, grad_key=grad_key)
        grads = self.compute_numeric_grads(
            params_array=data["params_array"],
            loss_array=data["loss_array"],
            adj_grad_array=data["adj_grad_array"],
        )

        valid_idx = set(map(int, grads["valid_indices"].tolist()))
        numeric_grads = grads["numeric_grad_array"]

        out_path = self.path(out_filename)
        with open(self.path(in_filename), "r") as fin, open(out_path, "w") as fout:
            i_numeric = 0
            for i_line, line in enumerate(fin):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if i_line in valid_idx:
                    rec["numeric_grad"] = numeric_grads[i_numeric].tolist()
                    rec["has_numeric_grad"] = True
                    i_numeric += 1
                else:
                    rec["has_numeric_grad"] = False
                fout.write(json.dumps(rec, default=self._json_convert) + "\n")

        if self.verbose:
            print(f"Augmented JSONL written: {out_path}")
        return out_path
