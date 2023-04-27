# Pipeline data_processing

## Overview

This pipeline processes the data. First the raw data is unzipped, then loaded using PartitionedDateset and finally it resized and turned into tensors 

## Pipeline outputs
* processed Dataset - dataset containg processed images, saved as pickle