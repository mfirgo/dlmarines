## Overview

This pipeline evaluates the model. It creates the Test Logger and DataModule, to make Test Trainer. It reads Trained Model saved by model_tarining pipeline and utilizes Test Triner to test it. 
![model_evaluation_visualization](./../../../../imgs/model_evaluation.png)

## Pipeline inputs

* PreprocessedDataset - dataset outputed by data_processing pipeline
* Training parameters -  parameters defining the training (for logging)
* logger parameters - parameters defining the wndb project for which to log


## Pipeline outputs

* Tested model - the model after testing

::: src.dlmarines.pipelines.model_evaluation.nodes