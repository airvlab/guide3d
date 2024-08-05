import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from guide3d.utils.utils import preprocess_tck
from torch.utils import data
from torchvision import transforms

image_transforms = transforms.Compose(
    [
        # transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]
)


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

            videoA.append(
                dict(
                    image=imageA,
                    tck=tckA,
                    u=uA,
                )
            )
            videoB.append(
                dict(
                    image=imageB,
                    tck=tckB,
                    u=uB,
                )
            )
        video_pairs.append(videoA)
        video_pairs.append(videoB)

    return video_pairs


def split_video_data(
    data: List,
    split: tuple = (0.8, 0.1, 0.1),
) -> List:
    train_data = []
    val_data = []
    test_data = []

    for video in data:
        train_idx = int(split[0] * len(video))
        val_idx = int(split[1] * len(video))
        train_data.extend(video[:train_idx])
        val_data.extend(video[train_idx : train_idx + val_idx])
        test_data.extend(video[train_idx + val_idx :])
    return train_data, val_data, test_data


class Guide3D(data.Dataset):
    """Guide3D dataset

    The dataset contains images and their corresponding t, c, k, u values,
    where:

    t: knot vector
    c: spline coefficients
    k: degree of the spline curve
    u: parameter values
    """

    def __init__(
        self,
        root: str,
        annotations_file: str = "sphere.json",
        image_transform: transforms.Compose = None,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
    ):
        self.root = Path(root)
        self.annotations_file = annotations_file
        raw_data = json.load(open(self.root / self.annotations_file))
        data = process_data(raw_data)
        train_data, val_data, test_data = split_video_data(data, split_ratio)
        assert split in [
            "train",
            "val",
            "test",
        ], "Split should be one of 'train', 'val', 'test'"

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = cv2.imread(str(self.root / sample["image"]), cv2.IMREAD_GRAYSCALE)
        t, c, k = sample["tck"]
        t = torch.tensor(t, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32)
        k = torch.tensor(k, dtype=torch.float32)

        u = torch.tensor(sample["u"], dtype=torch.float32)
        if self.image_transform:
            img = self.image_transform(img)
        return img, t, c, k, u


def visualize_mask(img, mask):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(mask, cmap="gray")
    plt.show()


def test_dataset():
    import guide3d.vars as vars

    dataset_path = vars.dataset_path
    dataset = Guide3D(dataset_path, "sphere_wo_reconstruct.json")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        img, t, c, k, u = batch
        print(img.shape, t.shape, c.shape, k.shape, u.shape)
        break


if __name__ == "__main__":
    test_dataset()