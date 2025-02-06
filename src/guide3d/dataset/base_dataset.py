from abc import ABC
from pathlib import Path
from typing import Optional, Tuple, Union

from torch.utils.data import Dataset

from guide3d.dataprocessor import DataProcessor
from guide3d.downloader import Downloader
from guide3d.vars import dataset_path


class BaseGuide3DDataset(Dataset, ABC):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        data_processor: DataProcessor,
        downloader: Optional[Downloader] = None,
        download: bool = False,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        augment: bool = False,
    ):
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.data_processor = data_processor
        self.downloader = downloader

        # Ensure dataset is downloaded if necessary
        if not self.dataset_path.exists():
            if download:
                if not self.downloader:
                    raise ValueError("Downloader is required when download=True")
                self.downloader.download(self.dataset_path)
            else:
                raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        self.data = self.data_processor.load_data(split, split_ratio)
        if augment:
            self.data = self.data_processor.apply_augmentation(self.data)

    def __len__(self) -> int:
        return len(self.data)


def main():
    from guide3d.dataprocessor import ImageDataProcessor
    from guide3d.downloader import GDriveDownloader

    # Create the processor (handles annotations automatically)
    data_processor = ImageDataProcessor(save_processed=True, annotation_filename="my_dataset.json")

    # Create a downloader (if needed)
    downloader = GDriveDownloader(file_id="1oRC_cQwGzrZ1XspPr9zwu6_rjWQE_kyI")

    # Create dataset instance
    dataset = BaseGuide3DDataset(
        dataset_path=dataset_path,
        data_processor=data_processor,
        downloader=downloader,
        download=False,
    )


if __name__ == "__main__":
    main()
