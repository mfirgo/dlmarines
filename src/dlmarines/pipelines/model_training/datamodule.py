import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from collections import defaultdict


class MarinesDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=32, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        class_to_id = defaultdict(lambda: len(class_to_id))
        dataset_new = []
        dataset = dict(dataset)
        for k, v in dataset.items():
            sample_class = k.split('/')[0]
            sample_class_id = class_to_id[sample_class]
            if v.shape[0] == 1:
                v = torch.vstack([v, v, v])
            elif v.shape[0] == 4:
                v = v[:3,:,:]
            dataset_new.append((v, sample_class_id))
        self.dataset = dataset_new
        self.id_to_class = {v: k for k, v in class_to_id.items()}
        self.split = False

    def split_dataset(self, fraction=0.2, seed=42):
        n = len(self.dataset)
        k = int(fraction*n)
        self.train, self.val, self.test = random_split(
            self.dataset, [n-2*k, k, k],
            generator=torch.Generator().manual_seed(seed)
        )
        self.split = True

    def setup(self, stage):
        if not self.split:
            self.split_dataset()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
    


