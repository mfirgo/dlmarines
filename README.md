# DLMarines 
**Team**: Bartosz Brzoza, Magdalena Buszka, Martyna Firgolska

**Description**: Image recognition of marine animals.

**Installation**:
```bash
conda env create  --file conda.yml
conda activate dlmarines
poetry install
```

**Download Kaggle Api Key**:
1. Sign in to [kaggle](https://www.kaggle.com/)
2. Go to Account
3. Go to API section and click `Create New API Token`. It will download `kaggle.json` with your username and key.
```json
{ "username":"your_kaggle_username","key":"123456789"}
```