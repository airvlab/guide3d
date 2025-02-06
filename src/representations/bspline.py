import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.special import comb
from sklearn.metrics import mean_squared_error


# Function to compute the residual sum of squares (RSS)
def compute_rss(points, bezier_points):
    return np.sum((points - bezier_points) ** 2)


def fit_spline(pts: np.ndarray, s: float = None, k: int = 3, eps: float = 1e-10):
    dims = pts.shape[1]
    if dims == 2:
        x = pts[:, 0]
        y = pts[:, 1]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + eps)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        tck, u = splprep([x, y], s=s, k=k, u=cumulative_distances)
        return tck, u
    elif dims == 3:
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2 + eps)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        tck, u = splprep([x, y, z], s=s, k=k, u=cumulative_distances)
        return tck, u
    else:
        raise ValueError("Input points must be 2D or 3D")


# Function to compute the Bernstein basis polynomials
def bernstein_basis(n, i, t):
    return comb(n, i) * (t**i) * ((1 - t) ** (n - i))


# Function to compute the Bézier curve points given control points
def bezier_curve(control_points, t_values):
    n = len(control_points) - 1
    curve = np.zeros((len(t_values), len(control_points[0])))
    for i in range(n + 1):
        b = bernstein_basis(n, i, t_values)
        curve += np.outer(b, control_points[i])
    return curve


# Function to fit a Bézier curve to a given set of points
def fit_bezier(points, degree=7):
    n = degree
    m = len(points)
    t_values = np.linspace(0, 1, m)

    # Build the matrix of Bernstein basis polynomials
    A = np.zeros((m, n + 1))
    for i in range(n + 1):
        A[:, i] = bernstein_basis(n, i, t_values)

    # Solve for the control points using least squares
    control_points_x = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]
    control_points_y = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
    control_points = np.column_stack((control_points_x, control_points_y))

    return control_points


# Function to find the best degree for the Bézier curve
# Function to find the best degree for fitting the Bézier curve
def find_best_degree(points, max_degree=10, mse_threshold=1e-3):
    best_degree = 1
    best_mse = float("inf")
    best_control_points = None
    t_values = np.linspace(0, 1, len(points))

    # Try fitting with increasing degrees
    for degree in range(1, max_degree + 1):
        control_points = fit_bezier(points, degree)
        bezier_points = bezier_curve(control_points, t_values)

        # Compute the mean squared error
        mse = mean_squared_error(points, bezier_points)

        # Update if the current degree is better
        if mse < best_mse:
            best_mse = mse
            best_degree = degree
            best_control_points = control_points

        # Stop if the error is low enough
        if best_mse < mse_threshold:
            break

    return best_degree, best_control_points, best_mse


def fit_piecewise_bezier(points, segment_count):
    n_points = len(points)
    segment_length = n_points // segment_count

    all_control_points = []
    piecewise_bezier_curves = []

    # For each segment, fit a cubic Bézier curve
    for i in range(segment_count):
        start = i * segment_length
        end = start + segment_length + 1
        if i == segment_count - 1:
            end = n_points  # Include all points in the last segment

        segment_points = points[start:end]
        degree = min(3, len(segment_points) - 1)  # Make sure degree is <= 3

        # Fit a cubic Bézier to this segment
        control_points = fit_bezier(segment_points, degree)
        all_control_points.append(control_points)

        # Reconstruct the curve for this segment
        t_values = np.linspace(0, 1, 100)
        bezier_segment = bezier_curve(control_points, t_values)
        piecewise_bezier_curves.append(bezier_segment)

    return np.vstack(piecewise_bezier_curves), all_control_points


def sample_spline(tck: tuple, u: list = None, n: int = None, delta: float = None):
    assert delta or n, "Either delta or n must be provided"
    assert not (delta and n), "Only one of delta or n must be provided"

    def is2d(tck):
        return len(tck[1]) == 2

    u_max = u[-1] if u is not None else tck[0][-1]
    num_samples = int(u_max / delta) + 1 if delta else n
    u = np.linspace(0, u_max, num_samples)
    if is2d(tck):
        x, y = splev(u, tck, ext=3)
        return np.column_stack([x, y]).astype(np.int32)
    else:
        x, y, z = splev(u, tck)
        return np.column_stack([x, y, z])


def main():
    import utils.viz as viz
    import vars
    from utils.utils import get_data

    dataset_path = vars.dataset_path
    data = get_data()

    fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(5, 10))
    axs = axs.flatten()
    plt.grid(True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
    plt.axis(False)
    for i, sample in enumerate(data[:6]):
        ax = axs[i]
        ax.axis("off")
        img = plt.imread(dataset_path / sample["img1"])
        pts = sample["pts1"]
        x = pts[:, 0]
        y = pts[:, 1]

        img = viz.convert_to_color(img)
        curve = fit_spline(pts)
        tck, u = curve
        control_points = tck[1]

        print("Knots", tck[0])
        print("Control Points", tck[1])
        print("Knots Len", len(tck[0]))
        print("Control Points Len", len(tck[1][0]))
        print("Number of control points:", len(tck[1][0]))

        x_fine, y_fine = splev(u, tck)
        ax.plot(x, y, "bo", label="Original points", **vars.plot_defaults)
        ax.plot(x_fine, y_fine, "g", label="Curve", **vars.plot_defaults)
        ax.plot(
            control_points[0],
            control_points[1],
            "yo",
            label="Control Points",
            **vars.plot_defaults,
        )
        ax.imshow(img)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        ncol=3,
        borderaxespad=0.1,
        handletextpad=0.1,
    )
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
