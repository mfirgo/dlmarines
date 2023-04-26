
from kedro.pipeline import Pipeline, node, pipeline
from dlmarines.pipelines.model_evaluation.nodes import test_model, get_trainer, get_logger, get_datamodule


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_datamodule,
                inputs="preprocessed_dataset",
                outputs="test_datamodule",
                name="get_test_datamodule_node"
            ),
            node(
                func=get_logger,
                inputs=["params:logger", "params:model"],
                outputs="test_logger",
                name="get_test_logger_node",
            ),
            node(
                func=get_trainer,
                inputs=["test_logger", "test_datamodule", "params:model_training"],
                outputs="test_trainer",
                name="get_test_trainer_node",
            ),
            node(
                func=test_model,
                inputs=["trained_model", "test_trainer", "test_datamodule"],
                outputs="tested_model",
                name="test_model_node",
            ),
        ]
    )
