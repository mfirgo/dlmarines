# DLMarines 
**Team**: Bartosz Brzoza, Magdalena Buszka, Martyna Firgolska

**Description**: Image recognition of marine animals.

## Installation:
```bash
conda env create  --file conda.yml
conda activate dlmarines
poetry install
```

## Data downloading
You can download data manually from https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste into [data/01_raw](./data/01_raw) or use data_downloading pipeline by running
```
kedro run --pipeline=data_downloading
```
Note that the pipeline uses kaggle api, so in order to run it follow the steps below to download your kaggle key.

**Download Kaggle Api Key**:
1. Sign in to [kaggle](https://www.kaggle.com/)
2. Go to Account
3. Go to API section and click `Create New API Token`. It will download `kaggle.json` with your username and key.
```json
{ "username":"your_kaggle_username","key":"123456789"}
```
4. In `conf\local\credentials.yml` add your username and key as shown below:
```yml
kaggle:
      username: "your_kaggle_username"
      key: "123456789"
```
## Data preprocesing
To preprocess data from [/data/01_raw/sea-animals-image-dataste.zip](./data/01_raw/sea-animals-image-dataste.zip) use data_processing pipeline
```
kedro run --pipeline=data_processing
```

## Running:
```
kedro run
```
