"""This script makes a JSON file with the annotations"""

import json

import cv2
import guide3d.calibration as calibration
import guide3d.utils.viz as viz
import guide3d.vars as vars
import matplotlib.pyplot as plt
import numpy as np
from guide3d.reconstruction import reconstruct
from guide3d.representations import curve
from guide3d.utils.fn import project_points
from parse_cvat import get_structured_dataset
from tqdm import tqdm

i = 0


def viz_bspline(img, tck, u):
    control_points = np.array(tck[1]).T
    sampled_pts = curve.sample_spline(tck, u, delta=0.01)

    # fig, ax = plt.subplots(1, 2, figsize=(5, 3))

    # set figure size
    plt.figure(figsize=(5, 5))

    # Plot the fitted Bézier curve
    plt.plot(sampled_pts[:, 0], sampled_pts[:, 1], "b-", label="Fitted Bézier Curve", linewidth=2)

    # Plot the control points and connect them with dashed lines
    plt.plot(control_points[:, 0], control_points[:, 1], "go--", label="Control Points", markersize=8)

    # Highlight control points with green circles
    for i, cp in enumerate(control_points):
        plt.text(cp[0], cp[1], f"P{i}", fontsize=12, color="green")

    # Label the axes and add a legend
    img = plt.imread(vars.dataset_path / img)
    plt.title("B-Spline Curve Fitting")
    plt.imshow(img, cmap="gray")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.savefig("bspline.png")
    # plt.grid(True)

    plt.show()
    plt.close()


def viz_curve(img, tck, u, pts=None, show=False):
    img = plt.imread(vars.dataset_path / img)
    pts = curve.sample_spline(tck, u, delta=10)

    plt.imshow(img, cmap="gray")
    plt.plot(tck[1][0], tck[1][1], "bo")
    plt.plot(pts[:, 0], pts[:, 1], "r")
    plt.show()
    return

    cv2.polylines(img, [pts], isClosed=False, color=255)

    new_h = int(img.shape[0] * 0.5)
    new_w = int(img.shape[1] * 0.5)

    if show:
        cv2.imshow("img", cv2.resize(img, (new_w, new_h)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def remove_close_points(pts, delta):
    cleaned_pts = [pts[0]]  # Start with the first point

    len_before = len(pts)
    for i in range(1, len(pts)):
        prev_point = cleaned_pts[-1]
        current_point = pts[i]
        distance = np.linalg.norm(np.array(prev_point) - np.array(current_point))

        if distance >= delta:
            cleaned_pts.append(current_point)

    len_after = len(cleaned_pts)
    # print(f"Removed {len_before - len_after} points")
    return cleaned_pts


def viz_curve_w_reprojection(imgA, imgB, tckA, tckB, tck3d, uA, uB, u3d, originalA, originalB, show=False):
    viz_path = imgA.split("/")[0]
    viz_path = viz_path.split("-")[:-1]
    viz_path = "-".join(viz_path)
    viz_path = vars.viz_dataset_path / "reprojection" / viz_path

    if not viz_path.exists():
        viz_path.mkdir(parents=True, exist_ok=True)

    img_num = imgA.split("/")[-1].split(".")[0]

    imgA = plt.imread(vars.dataset_path / imgA)
    imgB = plt.imread(vars.dataset_path / imgB)

    imgA = viz.convert_to_color(imgA)
    imgB = viz.convert_to_color(imgB)

    ptsA = curve.sample_spline(tckA, uA, delta=20)
    ptsB = curve.sample_spline(tckB, uB, delta=20)

    _, cA, _ = tckA
    _, cB, _ = tckB

    pts3d = curve.sample_spline(tck3d, u3d, delta=0.1)
    ptsA_reprojected = project_points(pts3d, calibration.P1)
    ptsB_reprojected = project_points(pts3d, calibration.P2)
    # print("len reprojected", len(ptsA_reprojected), len(ptsB_reprojected))

    # cv2.imshow(
    #     "imgA",
    #     cv2.resize(viz.draw_polyline(imgB, ptsB, color=(0, 255, 0)), (512, 512)),
    # )
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    fig, axs = plt.subplots(1, 2, figsize=(5, 3), sharex=True, sharey=True)
    plot_defaults = {"markersize": 0.4, "linewidth": 0.7, "alpha": 0.6}
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(imgA, cmap="gray")
    axs[1].imshow(imgB, cmap="gray")

    axs[0].plot(cA[0], cA[1], "yo", label="control points", **plot_defaults)
    axs[1].plot(cB[0], cB[1], "yo", label="control points", **plot_defaults)

    axs[0].plot(ptsA[:, 0], ptsA[:, 1], "g", label="original curve", **plot_defaults)
    axs[1].plot(ptsB[:, 0], ptsB[:, 1], "g", label="original curve", **plot_defaults)

    axs[0].plot(
        ptsA_reprojected[:, 0],
        ptsA_reprojected[:, 1],
        "r",
        label="reprojected",
        **plot_defaults,
    )
    axs[1].plot(
        ptsB_reprojected[:, 0],
        ptsB_reprojected[:, 1],
        "r",
        label="reprojected",
        **plot_defaults,
    )

    axs[0].plot(
        originalA[:, 0],
        originalA[:, 1],
        "bo",
        label="original points",
        **plot_defaults,
    )
    axs[1].plot(
        originalB[:, 0],
        originalB[:, 1],
        "bo",
        label="original points",
        **plot_defaults,
    )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        ncol=2,
        borderaxespad=0.1,
        handletextpad=0.1,
    )
    fig.savefig(viz_path / f"{img_num}.png", pad_inches=0, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    pass


def extract_properties(name: str) -> (str, str):
    folder, frame_number = name.split("/")
    fluid, task, guidewire_type, video_number, camera_number = folder.split("-")

    frame_number = frame_number.split(".")[0]

    return dict(
        fluid=int(fluid),
        task=task,
        guidewire_type=guidewire_type,
        video_number=int(video_number),
        camera_number=int(camera_number),
        frame_number=int(frame_number),
    )


def parse_into_dict(annotations: list) -> dict:
    dataset = {}
    for annotation in annotations:
        properties = extract_properties(annotation["image1"])
        properties.pop("camera_number")

        frame_number = properties.pop("frame_number")

        dataset_key = "-".join(str(value) for value in properties.values())
        dataset.setdefault(dataset_key, {})
        dataset[dataset_key][frame_number] = dict(frame_number=frame_number, **annotation)

    return dataset


def parse_key(key: str) -> dict:
    fluid, task, guidewire_type, video_number = key.split("-")
    return dict(
        fluid=int(fluid),
        guidewire_type=guidewire_type,
        video_number=int(video_number),
    )


def reorder_points(points: list) -> list:
    pt0 = points[0]
    pt_last = points[-1]
    if pt0[0] < pt_last[0]:
        points = points[::-1]
    return points


def make_json(dataset: dict, with_reconstruction: bool = False):
    root = []

    for key, value in dataset.items():
        properties = parse_key(key)

        video_pair = {}
        video_pair["fluid"] = properties["fluid"]
        video_pair["guidewire_type"] = properties["guidewire_type"]
        video_pair["video_number"] = properties["video_number"]
        video_pair["frame_count"] = len(value)
        video_pair["task"] = key

        frames = []
        for frame_number, annotation in value.items():
            frame = {
                "frame_number": frame_number,
                "camera2": {
                    "image": annotation["image1"],
                    "points": reorder_points(annotation["points1"].tolist()),
                },
                "camera1": {
                    "image": annotation["image2"],
                    "points": reorder_points(annotation["points2"].tolist()),
                },
            }

            frames.append(frame)
        video_pair["frames"] = frames
        root.append(video_pair)

    return root


def decompose_tck(tck):
    assert isinstance(tck, list) or isinstance(tck, tuple), f"tck should be a tuple or list, but got {type(tck)}\n"

    t, c, k = tck
    assert isinstance(t, np.ndarray), f"t should be a numpy array, but got {type(t)}\n"
    assert isinstance(k, int), f"k should be an integer, but got {type(k)}\n"

    t = t.tolist()
    c = [c_i.tolist() for c_i in c]

    return t, c, k


def visualize_bezier(control_points, img):
    # Reconstruct the Bezier curve from the control points
    t_values = np.linspace(0, 1, 100)  # Parameter values for the smooth curve
    bezier_points = curve.bezier_curve(control_points, t_values)
    # Visualization
    plt.figure(figsize=(8, 6))

    # Plot the fitted Bézier curve
    plt.plot(bezier_points[:, 0], bezier_points[:, 1], "b-", label="Fitted Bézier Curve", linewidth=2)

    # Plot the control points and connect them with dashed lines
    plt.plot(control_points[:, 0], control_points[:, 1], "go--", label="Control Points", markersize=8)

    # Highlight control points with green circles
    for i, cp in enumerate(control_points):
        plt.text(cp[0], cp[1], f"P{i}", fontsize=12, color="green")

    # Label the axes and add a legend
    img = plt.imread(vars.dataset_path / img)
    plt.title(f"Bézier Curve Fitting (Degree {len(control_points) - 1})")
    plt.imshow(img, cmap="gray")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()
    plt.close()


def make_annot_file(dataset: dict, with_reconstruction: bool = False):
    root = []
    valid_frames = 0

    for key, value in tqdm(dataset.items()):
        properties = parse_key(key)

        video_pair = {}
        video_pair["fluid"] = properties["fluid"]
        video_pair["guidewire_type"] = properties["guidewire_type"]
        video_pair["video_number"] = properties["video_number"]
        video_pair["frame_count"] = len(value)
        video_pair["task"] = key

        frames = []
        for frame_number, annotation in tqdm(value.items()):
            imageA = annotation["image1"]
            imageB = annotation["image2"]
            ptsA = reorder_points(annotation["points1"].tolist())
            ptsB = reorder_points(annotation["points2"].tolist())
            ptsA = remove_close_points(ptsA, 2)
            ptsB = remove_close_points(ptsB, 2)
            ptsA = np.clip(ptsA, 0, 1024)
            ptsB = np.clip(ptsB, 0, 1024)
            originalA = np.copy(ptsA)
            originalB = np.copy(ptsB)

            ptsA = np.array(ptsA)
            ptsB = np.array(ptsB)

            tckA, uA = curve.fit_spline_2(ptsA)
            tckB, uB = curve.fit_spline_2(ptsB)
            print(tckA)

            viz_curve(imageA, tckA, uA, ptsA, show=True)
            # viz_curve(imageB, tckB, uB, ptsB)

            # needed for JSON
            tA, cA, kA = decompose_tck(tckA)
            tB, cB, kB = decompose_tck(tckB)

            frame = {
                "frame_number": frame_number,
                "cameraA": {
                    "image": imageA,
                    "tck": {
                        "t": tA,
                        "c": cA,
                        "k": kA,
                    },
                    "u": uA.tolist(),
                },
                "cameraB": {
                    "image": imageB,
                    "tck": {
                        "t": tB,
                        "c": cB,
                        "k": kB,
                    },
                    "u": uB.tolist(),
                },
            }

            if with_reconstruction:
                tck3d, u3d = reconstruct(tckA, tckB, uA, uB, delta=20)
                if tck3d is None:
                    continue
                t3d, c3d, k3d = decompose_tck(tck3d)
                u3d = u3d.tolist()
                frame["3d"] = {
                    "tck": {
                        "t": t3d,
                        "c": c3d,
                        "k": k3d,
                    },
                    "u": u3d,
                }
                # check if there are too few points
                if tck3d[1][0].shape[0] < 4:
                    continue
                valid_frames += 1
                viz_curve_w_reprojection(
                    imageA,
                    imageB,
                    tckA,
                    tckB,
                    tck3d,
                    uA,
                    uB,
                    u3d,
                    originalA,
                    originalB,
                    show=False,
                )

            frames.append(frame)
        video_pair["frames"] = frames
        root.append(video_pair)
    print(f"Valid frames: {valid_frames}")

    exit()
    return root


def main():
    dataset = get_structured_dataset("data/annotations/cvat.xml")

    dataset = parse_into_dict(dataset)

    # json_data = make_json(dataset)
    # with open("data/annotations/raw.json", "w") as f:
    #     json.dump(json_data, f, indent=2)

    json_data = make_annot_file(dataset, with_reconstruction=False)
    with open("data/annotations/sphere_wo_reconstruct_bezier.json", "w") as f:
        json.dump(json_data, f, indent=2)

    #
    # with open("data/annotations/3d.json", "w") as f:
    #     json_data = make_json(dataset)
    #     json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    main()
