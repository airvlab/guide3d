import json
from pathlib import Path

import numpy as np
from parse_cvat import get_structured_dataset

from representations import curve


def remove_close_points(pts, delta=20):
    """Remove points that are closer than 'delta' to the previous point."""
    pts = pts.tolist() if isinstance(pts, np.ndarray) else pts
    if not pts:
        return pts
    cleaned = [pts[0]]
    for pt in pts[1:]:
        if np.linalg.norm(np.array(cleaned[-1]) - np.array(pt)) >= delta:
            cleaned.append(pt)
    return cleaned


def decompose_tck(tck):
    """Convert spline (tck) parameters into JSON-serializable lists."""
    t, c, k = tck
    return {"t": t.tolist(), "c": [ci.tolist() for ci in c], "k": k}


def process_annotation(ann):
    """
    For a given annotation, process both cameras:
      - Remove close points.
      - Fit a spline.
      - Decompose the spline parameters.
    """
    imageA = ann["image1"]
    imageB = ann["image2"]

    ptsA = remove_close_points(ann["points1"].tolist(), delta=20)
    ptsB = remove_close_points(ann["points2"].tolist(), delta=20)

    ptsA = np.array(ptsA)
    ptsB = np.array(ptsB)

    tckA, uA = curve.fit_spline(ptsA)
    tckB, uB = curve.fit_spline(ptsB)

    return {
        "cameraA": {
            "image": imageA,
            "tck": decompose_tck(tckA),
            "u": uA.tolist(),
        },
        "cameraB": {
            "image": imageB,
            "tck": decompose_tck(tckB),
            "u": uB.tolist(),
        },
    }


def main():
    # Get annotations from the CVAT XML file.
    annotations = get_structured_dataset("data/annotations/raw/cvat.xml")

    # Group annotations by their folder (task) extracted from image1.
    dataset = {}
    for ann in annotations:
        task = ann["image1"].split("/")[0]
        dataset.setdefault(task, []).append(ann)

    # Build JSON data.
    json_data = []
    for task, anns in dataset.items():
        task_data = {"task": task, "frame_count": len(anns), "frames": []}
        for idx, ann in enumerate(anns):
            frame_entry = {"frame_number": idx}
            frame_entry.update(process_annotation(ann))
            task_data["frames"].append(frame_entry)
        json_data.append(task_data)

    # Save the JSON output.
    output_file = Path("data/annotations/raw/raw.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved annotations to {output_file}")


if __name__ == "__main__":
    main()
