import pytorch_lightning as pl
from .model import MarineModel
from .datamodule import MarinesDataModule
from .callbacks import LogPredictionSamplesCallback


def get_datamodule(dataset):
    """Get marine datamodule for given dataset

    Args:
        dataset (PartitionedDataSet): dataset to be transformed into datamodule

    Returns:
        MarinesDataModule: datamodule with marine dataset data
    """
    datamodule = MarinesDataModule(dataset)
    return datamodule

def create_model(params):
    """Create Marine Model

    Args:
        params (dict): parameters for MarineModel

    Returns:
        MarineModel: untrained model
    """
    model = MarineModel(params)
    return model

def get_logger(logger_params, model_params):
    """Returns WandB logger

    Args:
        logger_params (dict): parameters for logger. 
            (Contains 'project_name' and 'entity_name' for WandB logger)
        model_params (_type_): parameters for model being trained.
            These parameters are loged so that different configurations can be compared.

    Returns:
        WandbLogger: logger
    """
    logger = pl.loggers.WandbLogger(
        project=logger_params['project_name'],
        entity=logger_params['entity_name'],
        config=model_params,
        log_model=True,
    )
    return logger

def get_trainer(logger, datamodule, params):
    """Get the trainer

    Args:
        logger (Logger): logger
        datamodule (MarinesDataModule): datamodule, used to get id to class name mapping
        params (dict): additional parameters for trainer (eg. num_epochs)

    Returns:
        Trainer: pytorch Lighting trainer
    """
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
    """Trains the model using provided trainer and datamodule

    Args:
        model (MarineModel): model to be trained
        trainer (Trainer): pytorch trainer to be used
        datamodule (MarinesDataModule): datamodule with data for training

    Returns:
        MarineModel: trained model
    """
    trainer.fit(
        model,
        datamodule=datamodule
    )
    return model


