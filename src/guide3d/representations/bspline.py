import json
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import BSpline as ScipyBSpline
from scipy.interpolate import make_lsq_spline


class BSpline:
    def __init__(self, pts: np.ndarray, delta: Union[int, float] = 30, k: int = 3) -> None:
        assert isinstance(pts, np.ndarray), f"pts should be np.ndarray but got: {type(pts)}"
        assert pts.shape[1] in {2, 3}, "Input points must be 2D or 3D"

        self.k = k
        self.delta = delta
        self.pts = pts

        # Define uniform parameterization (u values from 0 to 1)
        distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        u = np.insert(np.cumsum(distances), 0, 0)
        u /= u[-1]  # Normalize to [0,1]

        # Define uniform knots based on delta
        self.n = int(1 / delta)  # Number of segments
        self.t = np.concatenate(
            (
                np.zeros(k),  # k repetitions of 0 at start
                np.linspace(0, 1, self.n - k + 2),  # Uniformly spaced knots
                np.ones(k),  # k repetitions of 1 at end
            )
        )

        # Compute control points dynamically
        self.control_pts = self._compute_control_points(u, pts)

        # Construct B-Spline
        self.spline = ScipyBSpline(self.t, self.control_pts, self.k)

    def _compute_control_points(self, u: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Computes control points using a least-squares fit."""
        # Ensure enough points are present
        assert len(pts) >= self.k + 1, "Not enough points to fit the B-Spline."

        # Solve for control points using least squares
        return make_lsq_spline(u, pts, self.t, self.k).c

    def sample(self, n: int = 100) -> np.ndarray:
        """Samples points from the B-spline."""
        u_sample = np.linspace(0, 1, n)
        sampled_pts = self.spline(u_sample)
        return np.array(sampled_pts).T.astype(np.float64)

    def to_tensor(self) -> torch.Tensor:
        """Converts control points to a PyTorch tensor."""
        return torch.tensor(self.control_pts, dtype=torch.float32)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, delta: float, k: int = 3) -> "BSpline":
        """Creates a B-Spline from a PyTorch tensor of control points."""
        return cls(tensor.numpy(), delta=delta, k=k)

    def to_json(self) -> str:
        """Serializes the B-Spline control points to JSON."""
        data: Dict[str, Any] = {
            "control_pts": self.control_pts.tolist(),
            "t": self.t.tolist(),
            "k": self.k,
            "delta": self.delta,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BSpline":
        """Deserializes a B-Spline from JSON."""
        data = json.loads(json_str)
        obj = cls(np.zeros((len(data["control_pts"]), len(data["control_pts"][0]))), delta=data["delta"], k=data["k"])
        obj.control_pts = np.array(data["control_pts"])
        obj.t = np.array(data["t"])
        obj.spline = ScipyBSpline(obj.t, obj.control_pts, obj.k)
        return obj

    def plot(self, n: int = 100, ax: Optional[Any] = None, **plot_kwargs) -> None:
        """Plots the B-Spline curve and its control points."""
        sampled_pts = self.sample(n=n)

        fig_size = (6, 6)

        if ax is None:
            fig = plt.figure(figsize=fig_size)
            ax = fig.add_subplot(projection="3d") if sampled_pts.shape[1] == 3 else fig.add_subplot()

        if sampled_pts.shape[1] == 2:
            ax.plot(sampled_pts[:, 0], sampled_pts[:, 1], label="B-Spline Curve", **plot_kwargs)
            ax.scatter(self.control_pts[:, 0], self.control_pts[:, 1], color="red", marker="o", label="Control Points")
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.plot(sampled_pts[:, 0], sampled_pts[:, 1], sampled_pts[:, 2], label="B-Spline Curve", **plot_kwargs)
            ax.scatter(
                self.control_pts[:, 0],
                self.control_pts[:, 1],
                self.control_pts[:, 2],
                color="red",
                marker="o",
                label="Control Points",
            )

        ax.legend()
        plt.show()


if __name__ == "__main__":
    pts = np.random.rand(10, 2)
    spline = BSpline(pts, delta=0.01)

    # Sample & visualize
    spline.plot()

    # Convert to tensor & JSON
    tensor_pts = spline.to_tensor()
    json_str = spline.to_json()
    print(json_str)

    # Load from JSON
    spline2 = BSpline.from_json(json_str)
