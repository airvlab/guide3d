import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class DataProcessor(ABC):
    def __init__(
        self,
        save_processed: bool = True,
        annotation_filename: str = None,
        force_reprocess: bool = False,
    ):
        self.save_processed = save_processed
        self.force_reprocess = force_reprocess
        self.annotation_dir = Path(__file__).parent.joinpath("annotations").expanduser().resolve()

        self.processed_annotations_file = (
            self.annotation_dir.joinpath("processed", annotation_filename) if annotation_filename is not None else None
        )
        self.raw_annotations_file = self.annotation_dir.joinpath("raw/raw.json")

    def load_data(self, split: str, split_ratio: Tuple[float, float, float]) -> List:
        """Loads processed data if available; otherwise, processes and saves new data, then splits it."""

        if self.processed_annotations_file and not self.force_reprocess and self.processed_annotations_file.exists():
            self.data = self._load_json(self.processed_annotations_file)
        elif self.raw_annotations_file.exists():
            raw_data = self._load_json(self.raw_annotations_file)
            self.data = self.process_data(raw_data)

            if self.save_processed and self.processed_annotations_file:
                self._save_json(self.data, self.processed_annotations_file)
        else:
            raise FileNotFoundError(
                f"No annotations found in {self.raw_annotations_file}"
                + (f" or {self.processed_annotations_file}" if self.processed_annotations_file else "")
            )

        return self.split_data(self.data, split, split_ratio)

    @abstractmethod
    def process_data(self, raw_data: List[Dict]) -> List:
        """Processes raw data into a usable format. Includes optional augmentation."""
        pass

    @abstractmethod
    def split_data(self, data: List, split: str, split_ratio: Tuple[float, float, float]) -> List:
        """Splits the dataset into train, validation, and test sets."""
        pass

    @abstractmethod
    def apply_augmentation(self, data: List) -> List:
        pass

    def _load_json(self, path: Path) -> List:
        """Loads JSON data from a file."""
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found at: {path.as_posix()}")
        with open(path, "r") as f:
            return json.load(f)

    def _save_json(self, data: List, path: Path):
        """Saves processed data as JSON to avoid redundant processing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


class ImageDataProcessor(DataProcessor):
    """Default implementation of DataProcessor with optional augmentation."""

    def process_data(self, raw_data: List[Dict]) -> List:
        """Processes raw data and optionally applies augmentation."""
        frames = []
        for video_pair in raw_data:
            for frame in video_pair["frames"]:
                imageA = frame["cameraA"]["image"]
                imageB = frame["cameraB"]["image"]

                ptsA = frame["cameraA"]["points"]
                ptsB = frame["cameraB"]["points"]

                frames.append({"image": imageA, "points": ptsA})
                frames.append({"image": imageB, "points": ptsB})

        return frames

    def apply_augmentation(self, data: List[Dict]) -> List[Dict]:
        return data

    def split_data(self, data: List[Dict], split: str, split_ratio: Tuple[float, float, float]) -> List:
        """Splits the dataset into train, val, and test sets at the frame level."""
        np.random.shuffle(data)

        num_frames = len(data)
        train_idx = int(split_ratio[0] * num_frames)
        val_idx = train_idx + int(split_ratio[1] * num_frames)

        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]

        splits = {"train": train_data, "val": val_data, "test": test_data}
        return splits.get(split, [])


def main():
    data_processor = ImageDataProcessor(
        save_processed=True, force_reprocess=False, annotation_filename="my_dataset.json"
    )

    train_data = data_processor.load_data(split="train", split_ratio=(0.8, 0.1, 0.1))
    print(data_processor.get_statistics())
    data_processor.apply_augmentation(train_data)
    val_data = data_processor.load_data(split="val", split_ratio=(0.8, 0.1, 0.1))
    test_data = data_processor.load_data(split="test", split_ratio=(0.8, 0.1, 0.1))

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")


if __name__ == "__main__":
    main()
