
from ..model_training.nodes import get_trainer, get_datamodule, get_logger

def test_model(model, trainer, datamodule):
    """Tests the model

    Args:
        model (MarineModel): trained model.
        trainer (Trainer): trainer used for testing.
        datamodule (MarinesDataModule): datamodule with data for testing.

    Returns:
        MarineModel: tested model
    """
    trainer.test(
        model,
        datamodule=datamodule
    )
    return model
