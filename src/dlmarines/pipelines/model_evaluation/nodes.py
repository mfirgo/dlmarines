
def test_model(model, trainer, datamodule):
    trainer.test(
        model,
        datamodule=datamodule
    )
    return model

