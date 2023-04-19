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
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])


class MarinesDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
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
            # we do not need to change anything if number of channels is 3
            dataset_new.append((v, sample_class_id))
        self.dataset = dataset_new

    def setup(self, stage):
        n = len(self.dataset)
        k = int(0.2*n)
        self.train, self.val, self.test = random_split(self.dataset, [n-2*k, k, k])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=8)


def get_datamodule(dataset):
    datamodule = MarinesDataModule(dataset)
    return datamodule

def create_model(params):
    return MarineModel(params)

def get_logger(params):
    return pl.loggers.WandbLogger(
        project=params['project_name'],
    )

def get_trainer(logger, params):
    return pl.Trainer(
        max_epochs=params['num_epochs'],
        logger=logger
    )

def train_model(model, trainer, datamodule):
    trainer.fit(
        model,
        datamodule=datamodule
    )
    return model

def test_model(model, trainer, datamodule):
    trainer.test(
        model,
        datamodule=datamodule
    )
    return model


