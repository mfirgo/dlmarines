import os


def _setup_evironment_variables(kaggle_credentials):
    """Setups environment variables to be used with kaggle api.

    Args:
        kaggle_credentials (dict): Kaggle credentials. Has to contain fields:\n
            - 'KAGGLE_USERNAME' (string) - kaggle username
            - 'KAGGLE_KEY' (string) - kaggle api key
    """
    os.environ['KAGGLE_USERNAME'] = kaggle_credentials['username']
    os.environ['KAGGLE_KEY'] = kaggle_credentials['key']

def _connect_to_kaggle():
    """Imports and connects to kaggle api.

    Returns:
        KaggleApi object
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def download_dataset(kaggle_credentials: dict):
    """Downloads sea animals image dataset via kaggle api.
        
    Args:
        kaggle_credentials (dict): Kaggle credentials. Has to contain fields:\n
            - 'KAGGLE_USERNAME' (string) - kaggle username
            - 'KAGGLE_KEY' (string) - kaggle api key
    """
    _setup_evironment_variables(kaggle_credentials)
    api = _connect_to_kaggle()
    # typo in link is intentional
    api.dataset_download_files('vencerlanz09/sea-animals-image-dataste', path="./data/01_raw")