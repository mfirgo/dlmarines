"""
This is a boilerplate pipeline 'data_download'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline import Pipeline, node, pipeline
from kedro.config import ConfigLoader
from .nodes import print_conf, download_dataset


def create_pipeline(**kwargs) -> Pipeline:
    
    # conf_loader = ConfigLoader("conf/local")
    # params = conf_loader.get("credentials.yaml")
    print("from create pipeline {params}")
    return pipeline(
        [
            node(
                func=print_conf,
                inputs="params:kaggle",
                outputs=None,
                name="print_conf_node",
            ),
            node(
                func=download_dataset,
                inputs="params:kaggle",
                outputs=None,
                name="download_dataset_node",
            ),
        ]
    )