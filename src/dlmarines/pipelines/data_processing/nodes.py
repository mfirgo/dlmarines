import os
from kedro.io import PartitionedDataSet
import torchvision
from tqdm import tqdm
from zipfile import ZipFile

def unzip_dataset():
    with ZipFile('data/01_raw/sea-animals-image-dataste.zip', 'r') as f:
        f.extractall('data/01_raw')

    if not os.path.exists('data/01_raw/Corals/10712079_196e275866_o.jpg'):
        print('Please unzip the dataset:')
        print('data/01_raw/Corals/')
        print('data/01_raw/Crabs/')
        print('...')
    return 'unzipped'

def load_dataset(unzipped):
    dataset = PartitionedDataSet(
       path="data/01_raw/",
       dataset="pillow.ImageDataSet",
       filename_suffix=".jpg"
    )
    return dataset.load()

def preprocess_dataset(dataset):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean_vec, std_vec),
    ])
    transformed_dataset = dataset.copy()
    for k, v in tqdm(dataset.items()):
        transformed_dataset[k] = transform(v())
    return transformed_dataset

