
from ..model_training.nodes import get_trainer, get_datamodule, get_logger

def test_model(model, trainer, datamodule):
    trainer.test(
        model,
        datamodule=datamodule
    )
    return model
