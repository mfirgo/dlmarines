from kedro.pipeline import Pipeline, node, pipeline

from dlmarines.pipelines.data_processing.nodes import load_dataset, preprocess_dataset, unzip_dataset, make_class_id_dicts, train_test_split

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=unzip_dataset,
                inputs=None,
                outputs="unzipped_path",
                name="unzip_dataset_node",
            ),
            node(
                func=make_class_id_dicts,
                 inputs="unzipped_path",
                 outputs=["class_to_id_dict", "id_to_class_dict"],
                 name="make_class_to_id_dict_node" 
            ),
            node(
                func=load_dataset,
                inputs='unzipped_path',
                outputs="dataset",
                name="load_dataset_node",
            ),
            node(
                func=train_test_split,
                inputs="dataset",
                outputs=["train_dataset", "test_dataset"],
                name="train_test_split_node"
            ),
            node(
                func=preprocess_dataset,
                inputs="dataset",
                outputs="preprocessed_dataset",
                name="preprocess_dataset_node",
            ),
            # node(
            #     func=get_datamodule,
            #     inputs="preprocessed_dataset",
            #     outputs="datamodule", # train
            #     name="get_datamodule_node"
            # ),
            # node(
            #     func=get_datamodule,
            #     inputs="preprocessed_dataset",
            #     outputs="datamodule", # test
            #     name="get_datamodule_node"
            # ),
        ]
    )
