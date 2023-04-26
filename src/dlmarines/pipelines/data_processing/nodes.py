import os
from kedro.io import PartitionedDataSet
import torchvision
from tqdm import tqdm
from zipfile import ZipFile


def unzip_dataset():
    with ZipFile('data/01_raw/sea-animals-image-dataste.zip', 'r') as f:
        f.extractall('data/01_raw')
    
    return 'unzipped'

def load_dataset(unzipped):
    dataset = PartitionedDataSet(
       path="data/01_raw/",
       dataset="pillow.ImageDataSet",
       filename_suffix=".jpg"
    ).load()
    return dataset

def preprocess_dataset(dataset):
    transform = torchvision.transforms.Compose([
        # TODO: add size to params
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
    ])
    transformed_dataset = dataset.copy()

    for k, v in tqdm(dataset.items()):
        transformed_dataset[k] = transform(v())
    return transformed_dataset
