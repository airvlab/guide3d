from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import splprep
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

from guide3d.dataset.base_dataset import BaseGuide3DDataset
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


class Guide3D(BaseGuide3DDataset):
    """Guide3D dataset

    The dataset contains images and their corresponding t, c, u values,
    where:

    t: knot vector
    c: spline coefficients
    u: parameter values

    K, the degree of the spline, is 3.
    T, the knot vector, is of length n + k + 1, where n is the number of control
    points. The first k + 1 values are 0.
    """

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
        super(Guide3D, self).__init__(
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
    def fit_spline(pts: np.ndarray, s: float = None, k: int = 3, eps: float = 1e-10):
        if pts.shape[1] not in {2, 3}:
            raise ValueError("Input points must be 2D or 3D")

        # Compute cumulative distances
        deltas = np.diff(pts, axis=0)
        distances = np.sqrt((deltas**2).sum(axis=1) + eps)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

        # Fit spline
        tck, u = splprep(pts.T, s=s, k=k, u=cumulative_distances)
        return tck, u

    def process_data(self, raw_data: List[Dict]) -> List[List[Dict]]:
        video_pairs = []
        for video_pair in raw_data:
            videoA = []
            videoB = []
            for frame in video_pair["frames"]:
                imageA = frame["cameraA"]["image"]
                imageB = frame["cameraB"]["image"]

                ptsA = np.array(frame["cameraA"]["points"])
                ptsB = np.array(frame["cameraB"]["points"])

                tckA, uA = self.fit_spline(ptsA)
                tckB, uB = self.fit_spline(ptsB)

                videoA.append({"image": imageA, "tck": tckA, "u": uA.tolist()})
                videoB.append({"image": imageB, "tck": tckB, "u": uB.tolist()})

            video_pairs.append(videoA)
            video_pairs.append(videoB)

        return video_pairs

    @staticmethod
    def split_data(data: List[List[Dict]], split: str, split_ratio: Tuple[float, float, float]) -> List[Dict]:
        """
        Splits the processed video data into train, validation, and test sets.

        Args:
            data (List[List[Dict]]): Processed video data where each video is a list of frames.
            split (str): One of "train", "val", or "test".
            split_ratio (Tuple[float, float, float]): Ratios for train, val, and test splits.

        Returns:
            List[Dict]: The split data corresponding to the requested split type.
        """
        split_data = {"train": [], "val": [], "test": []}

        for video in data:
            num_frames = len(video)
            train_idx = int(split_ratio[0] * num_frames)
            val_idx = int(split_ratio[1] * num_frames)

            split_data["train"].extend(video[:train_idx])
            split_data["val"].extend(video[train_idx : train_idx + val_idx])
            split_data["test"].extend(video[train_idx + val_idx :])

        return split_data[split]

    @staticmethod
    def read_processed_annotations(path: Path) -> List:
        pass

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

    dataset = Guide3D(
        vars.dataset_path,
        image_transform=image_transform,
    )
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)

    print(len(dataset))
    batch = next(iter(dataloader))
    for batch in dataloader:
        img, target_seq, target_mask = batch
        sample = (img[0], target_seq[0], target_mask[0])

        # Visualize the sample
        Guide3D.visualize_sample(sample)


if __name__ == "__main__":
    main()
