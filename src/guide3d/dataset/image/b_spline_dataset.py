from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

import guide3d.vars as vars
from guide3d.dataprocessor import DataProcessor
from guide3d.dataset.base_dataset import BaseGuide3DDataset
from guide3d.downloader import Downloader
from guide3d.representations.bspline import BSplineCurve
from guide3d.representations.cubic import CubicSplineCurve
from guide3d.utils import sample_spline

IMAGE_SIZE = 1024
N_CHANNELS = 1

image_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(N_CHANNELS, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5 for _ in range(N_CHANNELS)],
            std=[0.5 for _ in range(N_CHANNELS)],
        ),
    ]
)


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
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(plt.imread(vars.dataset_path / imageA), cmap="gray")
                curveA.plot(50, ax)
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))
        curve = sample["curve"]
        spline = CubicSplineCurve.from_json(curve)
        spline = spline.to_tensor()

        t, c, _ = sample["tck"]

        t = torch.tensor(t[4:], dtype=torch.float32).unsqueeze(-1)
        c = np.array(c)
        c = torch.tensor(c.T, dtype=torch.float32)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(t), dtype=torch.int32)
        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


class Guide3DBSpline(BaseGuide3DDataset):
    k = 3

    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotation_file: Union[str, Path] = "b_spline_processed.json",
        image_transform: transforms.Compose = None,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        download: bool = False,
    ):
        super(Guide3DBSpline, self).__init__(
            dataset_path=dataset_path,
            annotation_file=annotation_file,
            save_processed=False,
            download=download,
            split=split,
            split_ratio=split_ratio,
        )

        self.image_transform = image_transform
        self.max_length = self._get_max_length()

    def _get_max_length(self):
        max_length = 0
        for video in self.all_data:
            for sample in video:
                t, c, _ = sample["tck"]
                max_length = max(max_length, len(t) - 4)
        return max_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))

        t, c, _ = sample["tck"]

        t = torch.tensor(t[4:], dtype=torch.float32).unsqueeze(-1)
        c = np.array(c)
        c = torch.tensor(c.T, dtype=torch.float32)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(t), dtype=torch.int32)
        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask

    @staticmethod
    def visualize_sample(sample, spline_sample_n=100):
        # Unpack the sample tuple.
        img, target_seq, target_mask = sample

        # --- 1. Denormalize the image ---
        # (Assuming image normalization with mean=0.5 and std=0.5.)
        img_denorm = img * 0.5 + 0.5

        # Convert image to a NumPy array.
        # For a grayscale image, squeeze out the channel dimension.
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)

        # --- 2. Extract valid control points ---
        # Use the target_mask to select only the valid rows from target_seq.
        valid_seq = target_seq[target_mask == 1]  # shape: (n_valid, D)
        valid_seq = valid_seq.cpu().numpy()

        t_vals = valid_seq[:, 0]
        c_vals = valid_seq[:, 1:]

        # We assume that the first column corresponds to t values (parameter values)
        # and the remaining columns are control point coordinates.
        zeros_to_add = 4  # Change this number if your spline degree (k) is different.
        full_t_vals = np.concatenate((np.zeros(zeros_to_add), t_vals))

        # --- 4. Construct the spline representation and sample it ---
        k = 3  # Spline degree.
        # Note: SciPy's splev expects the control points array in shape (dim, n_points),
        # so we transpose c_vals.
        tck = (full_t_vals, c_vals.T, k)
        spline_points = sample_spline(tck, n=spline_sample_n)

        # --- 4. Plotting ---
        plt.figure(figsize=(8, 8))

        # Display the image.
        if len(img_np.shape) == 2:
            plt.imshow(img_np, cmap="gray")
        else:
            plt.imshow(img_np)

        # Plot the control points.
        # Note: Depending on your coordinate system, you may need to flip axes.
        plt.scatter(c_vals[:, 0], c_vals[:, 1], c="red", s=50, label="Control Points")

        # Plot the spline curve.
        # Since sample_spline returns an array with shape (n_points, 2),
        # we treat the first column as x and the second as y.
        plt.plot(spline_points[:, 0], spline_points[:, 1], c="blue", linewidth=2, label="Spline Curve")

        plt.title("Guide3D Sample Visualization with Spline")
        plt.axis("off")
        plt.legend()
        plt.show()


def main():
    import guide3d.vars as vars

    dataset = Guide3DBSplineImageDataset(vars.dataset_path, True, force_preprocess=True)

    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    for batch in dataloader:
        img, target_seq, target_mask = batch
        sample = (img[0], target_seq[0], target_mask[0])

        # Visualize the sample
        Guide3DBSpline.visualize_sample(sample)


if __name__ == "__main__":
    main()
