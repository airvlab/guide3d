import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from parse_cvat import get_structured_dataset

import guide3d.vars as vars


def process_annotations(curve_class, dataset_path=None, visualize=False):
    """
    Processes annotations using the given curve class.

    Args:
        curve_class: A curve fitting class (e.g., BezierCurve, SplineCurve).
        dataset_path (str | Path): Path to the dataset (default: vars.dataset_path).
        visualize (bool): If True, visualize the fitted curves.
    """
    if dataset_path is None:
        dataset_path = vars.dataset_path

    dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path

    annotations = get_structured_dataset("data/annotations/cvat.xml")

    dataset = {}
    for ann in annotations:
        task = ann["image1"].split("/")[0]
        dataset.setdefault(task, []).append(ann)

    json_data = []
    for task, anns in dataset.items():
        task_data = {"task": task, "frame_count": len(anns), "frames": []}

        for idx, ann in enumerate(anns):
            frame_entry = {"frame_number": idx}

            imageA_path, imageB_path = ann["image1"], ann["image2"]
            ptsA, ptsB = np.array(ann["points1"]), np.array(ann["points2"])

            curveA, curveB = curve_class(ptsA), curve_class(ptsB)
            curveA.fit()
            curveB.fit()

            frame_entry["cameraA"] = {"image": imageA_path, "curve": curveA.to_dict()}
            frame_entry["cameraB"] = {"image": imageB_path, "curve": curveB.to_dict()}

            task_data["frames"].append(frame_entry)

            if visualize:
                imgA_full_path = dataset_path / imageA_path
                imgB_full_path = dataset_path / imageB_path

                imageA = plt.imread(imgA_full_path) if imgA_full_path.exists() else None
                imageB = plt.imread(imgB_full_path) if imgB_full_path.exists() else None

                curveA.visualize(image=imageA, img_size=512, title=f"{task} Frame {idx} - Camera A")
                curveB.visualize(image=imageB, img_size=512, title=f"{task} Frame {idx} - Camera B")

        json_data.append(task_data)

    # Save JSON
    output_file = dataset_path / f"annotations/{curve_class.__name__.lower()}_annotations.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {curve_class.__name__} annotations to {output_file}")
