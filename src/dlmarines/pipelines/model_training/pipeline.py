
from kedro.pipeline import Pipeline, node, pipeline
from dlmarines.pipelines.model_training.nodes import create_model, get_trainer, get_logger, train_model, get_datamodule


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_datamodule,
                inputs="preprocessed_dataset",
                outputs="datamodule",
                name="get_datamodule_node"
            ),
            node(
                func=get_logger,
                inputs="params:logger",
                outputs="logger",
                name="get_logger_node",
            ),
            node(
                func=get_trainer,
                inputs=["logger", "params:model_training"],
                outputs="trainer",
                name="get_trainer_node",
            ),
            node(
                func=create_model,
                inputs="params:model",
                outputs="model",
                name="create_model_node",
            ),
            node(
                func=train_model,
                inputs=["model", "trainer", "datamodule"],
                outputs="trained_model",
                name="train_model_node",
            ),
        ]
    )
