from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.io import read_image

from guide3d import vars
from guide3d.dataprocessor import DataProcessor
from guide3d.dataset.base_dataset import BaseGuide3DDataset
from guide3d.downloader import Downloader
from guide3d.normalizer import Normalizer
from guide3d.representations.bspline import BSplineCurve


class BSplineNormalizer(Normalizer):
    def __init__(self, t_max: int, c_max: int):
        self.t_min = 0
        self.t_max = t_max
        self.c_min = 0
        self.c_max = c_max

    def normalize(self, t: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes t-knots and c-values separately."""
        norm_t = (t - self.t_min) / (self.t_max - self.t_min + 1e-8)
        norm_c = (c - self.c_min) / (self.c_max - self.c_min + 1e-8)
        return norm_t, norm_c

    def unnormalize(self, norm_t: torch.Tensor, norm_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unnormalizes t-knots and c-values separately."""
        t = norm_t * (self.t_max - self.t_min) + self.t_min
        c = norm_c * (self.c_max - self.c_min) + self.c_min
        return t, c


class BSplineDataProcessor(DataProcessor):
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

    def compute_stats(self, data: List[Dict]) -> Dict[str, float]:
        """Computes dataset-wide statistics for normalization."""
        t_max, c_max = 0, 0
        for sample in data:
            curve = BSplineCurve.from_json(sample["curve"])
            t, c = curve.to_tensor()
            t_max = max(t_max, t.max().item())
            c_max = max(c_max, c.max().item())
        return {"t_max": t_max, "c_max": c_max}


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
        normalizer: Normalizer = None,
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
        self.normalizer = normalizer
        self.max_length = 25

    def get_stats(self):
        return {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))
        curve = sample["curve"]
        spline = BSplineCurve.from_json(curve)
        t, c = spline.to_tensor()
        t = t[4:].unsqueeze(-1)
        t, c = self.normalizer.normalize(t, c)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(t), dtype=torch.int32)
        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask

    def visualize_sample(self, sample):
        image, target_seq, target_mask = sample
        img, target_seq, target_mask = sample
        img_denorm = img * 0.5 + 0.5
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)

        valid_seq = target_seq[target_mask == 1]
        if valid_seq.shape[1] < 2:
            raise ValueError("Target sequence must contain at least (t, x, y) or (t, x, y, z).")

        t_vals = valid_seq[:, 0]
        t_vals = torch.cat((torch.zeros(4, device=t_vals.device, dtype=t_vals.dtype), t_vals))
        c_vals = valid_seq[:, 1:]
        t_vals, c_vals = self.normalizer.unnormalize(t_vals, c_vals)

        curve = BSplineCurve()
        curve.from_tensor(t=t_vals, c=c_vals, k=3)
        spline_points = curve.sample(100)

        plt.figure(figsize=(8, 8))

        if len(img_np.shape) == 2:
            plt.imshow(img_np, cmap="gray")
        else:
            plt.imshow(img_np)
        plt.scatter(c_vals[:, 0], c_vals[:, 1], c="red", s=10, label="Control Points")
        plt.plot(spline_points[:, 0], spline_points[:, 1], c="blue", linewidth=1, label="Spline Curve")
        plt.title("BSpline Visualization")
        plt.axis("off")
        plt.legend()
        plt.show()


def main():
    normalizer = BSplineNormalizer(1, 1024)
    dataset = Guide3DBSplineImageDataset(vars.dataset_path, True, force_preprocess=True, normalizer=normalizer)

    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    for batch in dataloader:
        img, target_seq, target_mask = batch
        sample = (img[0], target_seq[0], target_mask[0])
        dataset.visualize_sample(sample)


if __name__ == "__main__":
    main()
