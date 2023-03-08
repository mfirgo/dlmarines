from kedro.pipeline import Pipeline, node, pipeline

from dlmarines.pipelines.data_processing.nodes import load_dataset, preprocess_dataset, unzip_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=unzip_dataset,
                inputs=None,
                outputs="nothing",
                name="unzip_dataset_node",
            ),
            node(
                func=load_dataset,
                inputs=None,
                outputs="dataset",
                name="load_dataset_node",
            ),
            node(
                func=preprocess_dataset,
                inputs="dataset",
                outputs="preprocessed_dataset",
                name="preprocess_dataset_node",
            ),
            
        ]
    )
