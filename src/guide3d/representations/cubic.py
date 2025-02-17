import json
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline


class CubicSplineCurve:
    def __init__(self, pts: np.ndarray, uniform_param: bool = True):
        # pts: (N,2) or (N,3)
        if pts.ndim != 2 or pts.shape[1] not in {2, 3}:
            raise ValueError("pts must have shape (N,2) or (N,3)")
        self.pts = np.asarray(pts)
        N = self.pts.shape[0]
        if uniform_param:
            # Use a predefined, uniform parameterization along [0,1]
            self.u = np.linspace(0, 1, N)
        else:
            # Use chord-length parameterization (which is generally nonuniform)
            d = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
            cum = np.concatenate(([0], np.cumsum(d)))
            self.u = cum / cum[-1]
        self.spline = CubicSpline(self.u, self.pts, bc_type="not-a-knot")

    def sample(self, n: int = 100) -> np.ndarray:
        u_sample = np.linspace(0, 1, n)
        return self.spline(u_sample)

    def plot(self, n: int = 100, ax: Optional[plt.Axes] = None) -> None:
        pts_sample = self.sample(n)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        if pts_sample.shape[1] == 2:
            ax.plot(pts_sample[:, 0], pts_sample[:, 1], label="Cubic Spline")
            ax.scatter(self.pts[:, 0], self.pts[:, 1], color="red", label="Data Points")
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.plot(pts_sample[:, 0], pts_sample[:, 1], pts_sample[:, 2], label="Cubic Spline")
            ax.scatter(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2], color="red", label="Data Points")
        ax.legend()
        plt.show()

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.pts, dtype=torch.float32)

    def to_json(self) -> str:
        data = {"pts": self.pts.tolist(), "u": self.u.tolist()}
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "CubicSplineCurve":
        data = json.loads(json_str)
        pts = np.array(data["pts"])
        obj = cls(pts, uniform_param=False)  # We'll override u next.
        obj.u = np.array(data["u"])
        obj.spline = CubicSpline(obj.u, obj.pts, bc_type="not-a-knot")
        return obj


# --- Example Usage ---
if __name__ == "__main__":
    pts = np.array(
        [
            [3, 593],
            [100, 592],
            [192, 592],
            [302, 597],
            [346, 594],
            [374, 592],
            [407, 584],
            [432, 573],
            [451, 563],
            [469, 551],
            [503, 522],
            [524, 496],
            [537, 474],
            [551, 426],
            [554, 396],
            [549, 369],
            [539, 344],
            [525, 324],
            [514, 298],
        ]
    )
    # Create a cubic spline parametrized uniformly (i.e. at predefined points along [0,1])
    curve = CubicSplineCurve(pts, uniform_param=True)
    curve.plot()

    # Convert to JSON and restore.
    json_str = curve.to_json()
    curve_restored = CubicSplineCurve.from_json(json_str)
    curve_restored.plot()
