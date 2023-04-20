import os
from kedro.io import PartitionedDataSet
import torchvision
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from zipfile import ZipFile
import logging

logger = logging.getLogger(__name__)


def unzip_dataset():
    # check if data/01_raw/sea-animals exists
    if not os.path.exists("data/01_raw/sea-animals"):
        with ZipFile("data/01_raw/sea-animals-image-dataste.zip", "r") as f:
            f.extractall("data/01_raw/sea-animals")

        if not os.path.exists("data/01_raw/Corals/10712079_196e275866_o.jpg"):
            print("Please unzip the dataset:")
            print("data/01_raw/Corals/")
            print("data/01_raw/Crabs/")
            print("...")
    else:
        logger.info("Skipping unzip, data/01_raw/sea-animals already exists")
    return "data/01_raw/sea-animals"


def load_dataset(unzipped):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor(),
            # TODO: add preprocessing Normalization
        ]
    )
    return ImageFolder(
        unzipped,
        transform=transform
    )
