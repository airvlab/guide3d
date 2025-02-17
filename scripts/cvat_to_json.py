import json
from pathlib import Path

import numpy as np
from parse_cvat import get_structured_dataset


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

    ptsA = ensure_ordered_points(ptsA)
    ptsB = ensure_ordered_points(ptsB)

    return {
        "cameraA": {"image": imageA, "points": ptsA},
        "cameraB": {"image": imageB, "points": ptsB},
    }


def ensure_ordered_points(pts):
    """
    Ensure that points are ordered by increasing x coordinate.

    If the x coordinate of the second point is not greater than the first,
    return the list reversed.
    """
    pts = list(pts)
    if len(pts) < 2:
        return pts
    return pts if pts[0][0] < pts[-1][0] else pts[::-1]


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
    output_file = Path("data/annotations/raw/raw_2.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved annotations to {output_file}")


if __name__ == "__main__":
    main()
