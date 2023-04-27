## Data preprocesing
To preprocess data from `/data/01_raw/sea-animals-image-dataste.zip` use data_processing pipeline
```
kedro run --pipeline=data_processing
```
## Overview

This pipeline processes the data. First the raw data is unzipped, then loaded using PartitionedDateset and finally it resized and turned into tensors 

## Pipeline outputs
* processed Dataset - dataset containg processed images, saved as pickle

::: src.dlmarines.pipelines.data_processing.nodes