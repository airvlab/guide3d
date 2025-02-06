import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def parse_polyline(polyline_element: ET.Element) -> np.ndarray:
    points_str: str = polyline_element.get("points", "")
    points: np.ndarray = np.array(
        [list(map(float, point.split(","))) for point in points_str.split(";")],
        dtype=np.int32,
    )
    return points


def parse_image(image_element: ET.Element) -> Dict[str, Union[str, np.ndarray]]:
    name: str = image_element.get("name", "")
    polylines: List[ET.Element] = [parse_polyline(polyline) for polyline in image_element.findall("polyline")]
    points: Union[np.ndarray, None] = polylines[0] if polylines else None
    return {"name": name, "points": points}


def parse_annotation_file(
    file_path: Union[str, Path],
) -> List[Dict[str, Union[str, np.ndarray]]]:
    tree: ET.ElementTree = ET.parse(file_path)
    root: ET.Element = tree.getroot()
    return [parse_image(image) for image in root.findall("image")]


def extract_properties(name: str) -> (str, str):
    task, sample_number = name.split("/")
    camera_number: str = task[-1]
    task_name: str = task[:-2]
    identifier: str = f"{task_name}-{sample_number.split('.')[0]}"
    return identifier, camera_number


def pair_camera_annotations(
    annotations: List[Dict[str, Union[str, np.ndarray]]],
) -> List[Dict[str, Any]]:
    grouped_annotations: Dict[str, Dict[str, Any]] = {}
    for annot in annotations:
        identifier, camera_number = extract_properties(annot["name"])
        if identifier not in grouped_annotations:
            grouped_annotations[identifier] = {}
        grouped_annotations[identifier][f"image{camera_number}"] = annot["name"]
        grouped_annotations[identifier][f"points{camera_number}"] = annot["points"]
    # remove incomplete pairs
    grouped_annotations = {k: v for k, v in grouped_annotations.items() if len(v) == 4}
    return list(grouped_annotations.values())


def get_structured_dataset(
    annotation_file_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Union[str, np.ndarray]]] = parse_annotation_file(annotation_file_path)
    paired_annotations: List[Dict[str, Any]] = pair_camera_annotations(annotations)
    return paired_annotations


if __name__ == "__main__":
    import calibration
    import fn
    import reconstruct
    import vars

    dataset_path: Path = vars.dataset_path
    structured_dataset: List[Dict[str, Any]] = get_structured_dataset("data/annotations/cvat.xml")
    print(f"Number of paired annotations: {len(structured_dataset)}")

    lengths = []
    for entry in structured_dataset:
        points1 = entry["points1"]
        points2 = entry["points2"]

        p1, p2, p3d = reconstruct.get_points(points1, points2, calibration.P1, calibration.P2)
        length = fn.interpolate_even_spacing(p3d, spacing=0.17).shape[0]
        lengths.append(length)

    print(f"Minimum length: {min(lengths)}")
    print(f"Maximum length: {max(lengths)}")
