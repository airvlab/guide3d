import numpy as np
from base_curve import BaseCurve
from scipy.interpolate import splev, splprep


class SplineCurve(BaseCurve):
    """B-Spline representation using cumulative distance for parameterization."""

    def __init__(self, points, s=None, k=3, eps=1e-10):
        """
        Initialize a SplineCurve.

        Args:
            points (np.ndarray): Input points (N, 2) or (N, 3).
            s (float): Smoothing factor for splprep.
            k (int): Degree of the spline.
            eps (float): Small value to prevent zero distances.
        """
        super().__init__(points)
        self.s = s
        self.k = k
        self.eps = eps
        self.fitted_curve = None
        self.u = None

    def fit(self):
        """Fit a B-spline using cumulative distance parameterization."""
        if self.points.shape[0] < self.k + 1:
            raise ValueError(f"B-spline requires at least {self.k + 1} points.")

        dims = self.points.shape[1]

        if dims == 2:
            x = self.points[:, 0]
            y = self.points[:, 1]
            distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + self.eps)
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

            tck, u = splprep([x, y], s=self.s, k=self.k, u=cumulative_distances)
        elif dims == 3:
            x = self.points[:, 0]
            y = self.points[:, 1]
            z = self.points[:, 2]
            distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2 + self.eps)
            cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
            tck, u = splprep([x, y, z], s=self.s, k=self.k, u=cumulative_distances)

        self.fitted_curve, self.u = splprep(coords, s=self.s, k=self.k, u=cumulative_distances)

    def to_dict(self):
        """Convert spline to JSON-serializable format."""
        t, c, k = self.fitted_curve
        return {"type": "spline", "t": t.tolist(), "c": [ci.tolist() for ci in c], "k": k, "u": self.u.tolist()}

    def reconstruct(self, num_points=100):
        """Reconstruct the spline curve."""
        if self.fitted_curve is None:
            self.fit()
        u_new = np.linspace(self.u[0], self.u[-1], num_points)
        return np.array(splev(u_new, self.fitted_curve)).T


if __name__ == "__main__":
    from process_annotations import process_annotations

    process_annotations(SplineCurve, visualize=True)
