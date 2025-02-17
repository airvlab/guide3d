from torchvision import transforms


def vit_transforms(image_size: int = 1024, n_channels: int = 1):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(n_channels, 1, 1)),
            transforms.Normalize(mean=[0.5 for _ in range(n_channels)], std=[0.5 for _ in range(n_channels)]),
        ]
    )
