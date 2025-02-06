import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import gdown


class Downloader(ABC):
    """Abstract base class for downloaders."""

    @abstractmethod
    def download(self, output_path: Path):
        """Download a file to the given output path."""
        pass

    def cleanup(self, file_path: Path):
        """Remove a file after extraction or if no longer needed."""
        if file_path.exists():
            file_path.unlink()


class GDriveDownloader(Downloader):
    """Downloader for Google Drive files."""

    def __init__(self, file_id: str, quiet: bool = False):
        self.file_id = file_id
        self.quiet = quiet

    def download(self, output_path: Path):
        """Download and extract a ZIP file from Google Drive."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Define temporary zip file path
        zip_path = output_path / "temp.zip"

        # Download file
        gdown.download(id=self.file_id, output=str(zip_path), quiet=self.quiet)

        # Extract if it's a zip file
        if zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(output_path)

            # Cleanup temporary ZIP file
            self.cleanup(zip_path)


if __name__ == "__main__":
    downloader = GDriveDownloader(file_id="1oRC_cQwGzrZ1XspPr9zwu6_rjWQE_kyI")
    downloader.download(Path("test-data"))
