import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import BSpline, make_splprep


class BSplineCurve:
    def __init__(self):
        self.spline = None

    def fit(self, pts):
        pts = np.asarray(pts)
        self.original_pts = pts
        if pts.ndim != 2 or pts.shape[1] not in {2, 3}:
            raise ValueError("pts must have shape (N,2) or (N,3)")

        self.spline, self.knots = make_splprep(pts.T, s=1e-3)

    def sample(self, n=100):
        if self.spline is None:
            raise RuntimeError("Spline has not been fitted. Call `fit(pts)` first.")

        u_sample = np.linspace(0, 1, n)
        return self.spline(u_sample)

    def plot(self, n=100, ax=None):
        """Plots the fitted B-Spline curve."""
        if self.spline is None:
            raise RuntimeError("Spline has not been fitted. Call `fit(pts)` first.")
        control_points = self.spline.c
        samples = self.sample(n)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(samples[:, 0], samples[:, 1], label="B-Spline Curve")
        ax.scatter(control_points[:, 0], control_points[:, 1], color="red", marker="o", label="Control Points")
        ax.set_aspect("equal")
        ax.legend()
        plt.show()

    def overlay_on_image(self, image_path, n=100, color=(0, 0, 255), thickness=2):
        """Overlays the B-Spline curve on an image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image.")

        samples = self.sample(n).astype(int)

        for i in range(len(samples) - 1):
            cv2.line(image, tuple(samples[i]), tuple(samples[i + 1]), color, thickness)

        cv2.imshow("B-Spline Overlay", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def to_tensor(self):
        """Converts the B-Spline data to PyTorch tensors."""
        if self.spline is None:
            raise RuntimeError("Spline has not been fitted. Call `fit(pts)` first.")
        return torch.tensor(self.spline.t, dtype=torch.float32), torch.tensor(self.spline.c, dtype=torch.float32)

    def from_tensor(self, t: torch.Tensor, c: torch.Tensor, k: int):
        t = t.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        self.spline = BSpline(t, c, k)

    def to_json(self):
        if self.spline is None:
            raise RuntimeError("Spline has not been fitted. Call `fit(pts)` first.")
        return json.dumps(
            {
                "original_pts": self.original_pts.tolist(),
                "t": self.spline.t.tolist(),
                "c": self.spline.c.tolist(),
                "k": self.spline.k,
            }
        )

    @classmethod
    def from_json(cls, json_str):
        """Deserializes a JSON string into a BSplineCurve object and restores `self.spline`."""
        data = json.loads(json_str)
        obj = cls()
        obj.spline = BSpline(data["t"], data["c"], data["k"])
        obj.original_pts = data["original_pts"]
        return obj


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

    curve = BSplineCurve()
    curve.fit(pts)
    curve.plot()
    t, c = curve.to_tensor()
    curve_normal = curve.from_tensor(t, c, 3)
