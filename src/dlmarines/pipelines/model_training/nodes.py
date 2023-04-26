import pytorch_lightning as pl
from .model import MarineModel
from .datamodule import MarinesDataModule
from .callbacks import LogPredictionSamplesCallback


def get_datamodule(dataset):
    datamodule = MarinesDataModule(dataset)
    return datamodule

def create_model(params):
    model = MarineModel(params)
    return model

def get_logger(params):
    logger = pl.loggers.WandbLogger(
        project=params['project_name'],
        log_model=True,
    )
    return logger

def get_trainer(logger, params):
    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        # TODO: add lots of params
        logger=logger,
        callbacks=[
            LogPredictionSamplesCallback(),
        ]
    )
    return trainer

def train_model(model, trainer, datamodule):
    trainer.fit(
        model,
        datamodule=datamodule
    )
    return model


