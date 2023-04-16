import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from collections import defaultdict


class MarineModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        layers = []
        layers.append(
            nn.Conv2d(3, params['cnn_channels'], params['kernel_size'])
        )
        layers.append(nn.ReLU())
        for _ in range(params['num_cnn_layers']-1):
            layers.append(nn.Conv2d(params['cnn_channels'], params['cnn_channels'], params['kernel_size']))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        layers.append(nn.Linear(params['flattened_size'], params['fc_features']))
        for _ in range(params['num_fc_layers']-1):
            layers.append(nn.Linear(params['fc_features'], params['fc_features']))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(params['fc_features'], params['num_classes']))
        layers.append(nn.Softmax())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])


class MarinesDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        class_to_id = defaultdict(lambda: len(class_to_id))
        dataset_new = []
        for k, v in dataset.items():
            sample_class = k.split('/')[0]
            sample_class_id = class_to_id[sample_class]
            if v.shape[0] == 1:
                v = torch.vstack([v, v, v])
            elif v.shape[0] == 4:
                v = v[:3,:,:]
            # we do not need to change anything if number of channels is 3
            dataset_new.append((v, sample_class_id))
        self.dataset = dataset_new

    def setup(self, stage):
        n = len(self.dataset)
        k = int(0.8*n)
        if stage == "fit":
            self.train, self.val = random_split(self.dataset, [k, n-k])
        if stage == "test":
            self.test = self.dataset
        if stage == "predict":
            self.predict = self.dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=32)
    

def get_datamodules(dataset):
    n = len(dataset)
    k = int(0.8*n)
    train_dataset, test_dataset = random_split(dataset, [k, n-k])
    train_datamodule = MarinesDataModule(train_dataset)
    test_datamodule = MarinesDataModule(test_dataset)
    
    return train_datamodule, test_datamodule

def create_model(params):
    return MarineModel(params)

def get_trainer(params):
    return pl.Trainer(
        max_epochs=params['num_epochs']
    )

def train_model(model, trainer, train_datamodule):
    trainer.fit(
        model,
        datamodule=train_datamodule
    )
    return model

def test_model(model, trainer, test_datamodule):
    trainer.test(
        model,
        datamodule=test_datamodule
    )
    return model


