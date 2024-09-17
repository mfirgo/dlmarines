import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class MarineModel(pl.LightningModule):
    """MarineModel class. Inherits from pytorch LightningModule
    """
    def __init__(self, params):
        """Initializes model

        Args:
            params (dict): Dict of parameters described below
            params.cnn_channles(int): number of cnn channels for convolutional layer
            params.kernel_size(int): kernel size in convolutional layer
            params.num_cnn_layers: number of convolutional layers
            params.flattened_size: number of input neurons for first fully connected layer
            params.fc_features: number of neurons in fully connected layers
            params.num_fc_layers: number of fully conected layers
            params.num_classes: number of classes to be classified
        """
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
        layers.append(nn.ReLU())
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
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss, y_hat
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])


