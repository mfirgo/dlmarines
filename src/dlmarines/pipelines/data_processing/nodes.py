import os
from kedro.io import PartitionedDataSet
import torchvision
from tqdm import tqdm
from zipfile import ZipFile
# from kedro.extras


def unzip_dataset():
    unzipped_path = "data/01_raw/"
    with ZipFile('data/01_raw/sea-animals-image-dataste.zip', 'r') as f:
        f.extractall(unzipped_path)

    if not os.path.exists(unzipped_path + 'Corals/10712079_196e275866_o.jpg'):
        print('Please unzip the dataset:')
        print(unzipped_path + 'Corals/')
        print(unzipped_path + 'Crabs/')
        print('...')
    
    return unzipped_path


# def load_dataset(unzipped_path):
#     dataset = PartitionedDataSet(
#        path=unzipped_path,
#        dataset="pillow.ImageDataSet",
#        filename_suffix=".jpg"
#     )
#     return dataset.load()

def load_dataset(unzipped_path):
    dataset = PartitionedDataSet(
       path=unzipped_path,
       dataset="pillow.ImageDataSet",
       filename_suffix=".jpg"
    )
    return dataset.load()

def make_class_id_dicts(unzipped_path):
    classes_names = [f.path.split('/')[-1] for f in os.scandir(unzipped_path) if f.is_dir() ]
    class_to_id_dict = {class_name: i  for i, class_name in enumerate(classes_names)}
    id_to_class_dict = {i: class_name for class_name, i in class_to_id_dict.items()}
    return class_to_id_dict, id_to_class_dict

def preprocess_dataset(dataset):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda x: x.repeat(3,1,1)) #### tutaj trzeba wrzucić tego ifa z channelami jako funkcję pomocniczą i to dać zamiast lambdy
        # torchvision.transforms.Normalize(mean_vec, std_vec),
    ])
    transformed_dataset = dataset.copy()

    for k, v in tqdm(dataset.items()):
        transformed_dataset[k] = transform(v())
        print(k)
        print(v)
        break
    return transformed_dataset


# def train_test_split(dataset):

#     return train_dataset, test_dataset






