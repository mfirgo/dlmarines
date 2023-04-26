
from kedro.pipeline import Pipeline, node, pipeline
from dlmarines.pipelines.model_evaluation.nodes import test_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_model,
                inputs=["trained_model", "trainer", "datamodule"],
                outputs="tested_model",
                name="test_model_node",
            ),
        ]
    )
