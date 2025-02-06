import abc

import matplotlib.pyplot as plt
import numpy as np


class BaseCurve(abc.ABC):
    """Abstract base class for curve fitting and visualization."""

    def __init__(self, points):
        self.points = np.array(points)
        self.fitted_curve = None

    @abc.abstractmethod
    def fit(self):
        """Fit the curve to the given points."""
        pass

    @abc.abstractmethod
    def to_dict(self):
        """Convert curve parameters to JSON."""
        pass

    @abc.abstractmethod
    def reconstruct(self, num_points=100):
        """Reconstruct the curve."""
        pass

    def visualize(self, image=None, img_size=512, title="Fitted Curve"):
        """
        Plot the fitted curve on an image (if provided) or a blank background.

        Args:
            image (numpy.ndarray): Background image (optional).
            width (int): Width of the blank canvas if no image is provided.
            height (int): Height of the blank canvas if no image is provided.
            title (str): Plot title.
        """
        reconstructed = self.reconstruct()

        plt.figure(figsize=(6, 6))

        if image is not None:
            plt.imshow(image, cmap="gray", extent=[0, img_size, img_size, 0])

        plt.scatter(self.points[:, 0], self.points[:, 1], color="red", label="Original Points", s=20)
        plt.plot(reconstructed[:, 0], reconstructed[:, 1], color="blue", label="Fitted Curve", linewidth=2)

        plt.xlim(0, img_size)
        plt.ylim(img_size, 0)  # Flip Y-axis to match image coordinates
        plt.title(title)
        plt.legend()
        plt.show()
