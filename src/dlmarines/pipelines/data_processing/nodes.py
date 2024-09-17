import os
from kedro.io import PartitionedDataSet
import torchvision
from tqdm import tqdm
from zipfile import ZipFile


def unzip_dataset():
    """Unzips the dataset.

    Note that for this function to work correctly the sea animals dataset 
    should be already downloaded into `data/01_raw/sea-animals-image-dataste.zip` 
    either manually or via running data_downloading pipeline.

    Returns:
        string: 'unzipped'
    """
    with ZipFile('data/01_raw/sea-animals-image-dataste.zip', 'r') as f:
        f.extractall('data/01_raw')
    # TODO: return path to unzipped dataset
    return 'unzipped'

def load_dataset(unzipped):
    """Loades dataset as ImageDataSet.

    Args:
        unzipped (any): indicates that the previous node has been run

    Returns:
        PartitionedDataSet: dataset loaded from "data/01_raw/"
    """
    dataset = PartitionedDataSet(
       path="data/01_raw/",
       dataset="pillow.ImageDataSet",
       filename_suffix=".jpg"
    ).load()
    return dataset

def preprocess_dataset(dataset):
    """Preprocesses dataset. Each image in dataset is resized and transformed to tensor.

    Args:
        dataset (PartitionedDataSet): Loaded image dataset

    Returns:
        PartitionedDataSet: transformed dataset
    """
    transform = torchvision.transforms.Compose([
        # TODO: add size to params
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
    ])
    transformed_dataset = dataset.copy()

    for k, v in tqdm(dataset.items()):
        transformed_dataset[k] = transform(v())
    return transformed_dataset
