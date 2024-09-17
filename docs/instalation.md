## Installation:
To install this project make sure you have [conda](https://docs.anaconda.com/free/anaconda/install/) and [poetry](https://python-poetry.org/docs/) installed. After that simply open terminal, go to project folder and type:  


```bash
conda env create  --file conda.yml
conda activate dlmarines
poetry install
```

----

*Note that the download pipeline uses kaggle api, so in order to run it follow the steps below to download your kaggle key.*

----

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

