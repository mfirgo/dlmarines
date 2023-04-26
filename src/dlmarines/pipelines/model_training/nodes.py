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

def get_logger(logger_params, model_params):
    logger = pl.loggers.WandbLogger(
        project=logger_params['project_name'],
        entity=logger_params['entity_name'],
        config=model_params,
        log_model=True,
    )
    return logger

def get_trainer(logger, datamodule, params):
    trainer = pl.Trainer(
        max_epochs=params['num_epochs'],
        # TODO: add lots of params
        logger=logger,
        callbacks=[
            LogPredictionSamplesCallback(datamodule.id_to_class),
        ]
    )
    return trainer

def train_model(model, trainer, datamodule):
    trainer.fit(
        model,
        datamodule=datamodule
    )
    return model


