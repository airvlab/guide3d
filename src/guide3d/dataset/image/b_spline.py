from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.utils import preprocess_tck, sample_spline, split_fn_image

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


def visualize_sample(sample, spline_sample_n=100):
    """
    Visualize a single sample from the Guide3D dataset.

    The sample is expected to be a tuple:
      - img: Tensor of shape [C, H, W] (normalized using mean=0.5, std=0.5)
      - target_seq: Tensor of shape [max_length, D] where the first column is t
                    and the remaining columns are control coordinates c.
      - target_mask: Tensor of shape [max_length] (1 for valid entries, 0 for padding)

    This function:
      1. Denormalizes and converts the image for display.
      2. Extracts valid control points from target_seq.
      3. Constructs a tck tuple (with t as knot vector, c as control points, k=3),
         adjusts the shape of c as needed, and samples the spline using sample_spline.
      4. Overlays both the control points and the sampled spline curve.
    """
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


def process_data(
    data: Dict,
) -> List:
    video_pairs = []
    for video_pair in data:
        videoA = []
        videoB = []
        for frame in video_pair["frames"]:
            imageA = frame["cameraA"]["image"]
            imageB = frame["cameraB"]["image"]

            tckA = preprocess_tck(frame["cameraA"]["tck"])
            tckB = preprocess_tck(frame["cameraB"]["tck"])

            uA = np.array(frame["cameraA"]["u"]).astype(np.float32)
            uB = np.array(frame["cameraB"]["u"]).astype(np.float32)

            videoA.append(dict(image=imageA, tck=tckA, u=uA))
            videoB.append(dict(image=imageB, tck=tckB, u=uB))
        video_pairs.append(videoA)
        video_pairs.append(videoB)

    return video_pairs


class Guide3D(BaseGuide3D):
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
        annotations_file: Union[str, Path] = "b_spline.json",
        image_transform: transforms.Compose = None,
        c_transform: callable = None,
        t_transform: callable = None,
        transform_both: callable = None,
        batch_first: bool = False,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        download: bool = False,
    ):
        super(Guide3D, self).__init__(
            dataset_path=dataset_path,
            annotations_file=annotations_file,
            process_data=process_data,
            split_fn=split_fn_image,
            download=download,
            split=split,
            split_ratio=split_ratio,
        )

        self.image_transform = image_transform
        self.c_transform = c_transform
        self.t_transform = t_transform
        self.transform_both = transform_both
        self.max_length = self._get_max_length()

    def __len__(self):
        return len(self.data)

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

        # t has 4 zeros at the beginning
        t = torch.tensor(t[4:], dtype=torch.float32).unsqueeze(-1)
        c = torch.tensor(c, dtype=torch.float32)

        if self.transform_both:
            img, t, c = self.transform_both(img, t, c)

        if self.t_transform:
            t = self.t_transform(t)

        if self.c_transform:
            c = self.c_transform(c)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(t), dtype=torch.int32)
        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


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
        visualize_sample(sample)


if __name__ == "__main__":
    main()
