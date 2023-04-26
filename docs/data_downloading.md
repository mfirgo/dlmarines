
## Data downloading
You can download data manually from https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste into `data/01_raw` or use data_downloading pipeline by running
```
kedro run --pipeline=data_downloading
```
*Note that the pipeline uses kaggle api, so you have to download kaggle Api Key if you want to run it. You can find how to do it on [instalation section](instalation.md#Download-Kaggle-Api-Key)*

::: src.dlmarines.pipelines.data_downloading.nodes