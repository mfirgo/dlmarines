"""
This is a boilerplate pipeline 'data_download'
generated using Kedro 0.18.6
"""
from typing import Any

from icecream import ic
import os


def _setup_evironment_variables(kaggle_credentials: dict[str, Any]):
    ic("kaggle___RNO", kaggle_credentials)
    os.environ['KAGGLE_USERNAME'] = kaggle_credentials['KAGGLE_USERNAME']
    os.environ['KAGGLE_KEY'] = kaggle_credentials['KAGGLE_KEY']


def _connect_to_kaggle():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(kaggle_credentials: dict):
    _setup_evironment_variables(kaggle_credentials)
    api = _connect_to_kaggle()
    # api.dataset_download_files('vencerlanz09/sea-animals-image-dataste', path="./data/01_raw") # ???


def print_conf(my_parameters: dict):
    print(my_parameters)
    return my_parameters
