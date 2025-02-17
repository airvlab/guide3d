from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.io import read_image

import guide3d.vars as vars
from guide3d.dataprocessor import DataProcessor
from guide3d.dataset.base_dataset import BaseGuide3DDataset
from guide3d.downloader import Downloader
from guide3d.representations.bspline import BSplineCurve


class BSplineDataProcessor(DataProcessor):
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

                curveA = BSplineCurve()
                curveA.fit(ptsA)
                # fig, ax = plt.subplots(figsize=(6, 6))
                # ax.imshow(plt.imread(vars.dataset_path / imageA), cmap="gray")
                # curveA.plot(50, ax)
                curveB = BSplineCurve()
                curveB.fit(ptsB)

                frames.append({"image": imageA, "curve": curveA.to_json()})
                frames.append({"image": imageB, "curve": curveB.to_json()})

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


class Guide3DBSplineImageDataset(BaseGuide3DDataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        download: bool = False,
        downloader: Optional[Downloader] = None,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        augment: bool = False,
        force_preprocess: bool = False,
        image_transform: callable = None,
    ):
        data_processor = BSplineDataProcessor(
            save_processed=True,
            annotation_filename="b_spline_image_dataset.json",
            force_reprocess=force_preprocess,
        )

        super().__init__(
            dataset_path=dataset_path,
            data_processor=data_processor,
            downloader=downloader,
            download=download,
            split=split,
            split_ratio=split_ratio,
            augment=augment,
        )

        self.image_transform = image_transform
        self.max_length = 19

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))
        curve = sample["curve"]
        spline = BSplineCurve.from_json(curve)
        t, c = spline.to_tensor()
        t = t[4:].unsqueeze(-1)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(t), dtype=torch.int32)
        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def main():
    dataset = Guide3DBSplineImageDataset(vars.dataset_path, True, force_preprocess=False)

    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    exit()
    for batch in dataloader:
        img, target_seq, target_mask = batch
        sample = (img[0], target_seq[0], target_mask[0])


if __name__ == "__main__":
    main()
