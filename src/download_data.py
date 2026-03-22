import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract():
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    # Ensure the raw directory exists
    os.makedirs('../data/raw', exist_ok=True)
    
    print("Downloading dataset...")
    # Downloads the zip file into the data/raw directory
    api.competition_download_files('march-machine-learning-mania-2026', path='../data/raw')
    
    zip_path = '../data/raw/march-machine-learning-mania-2026.zip'
    
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('../data/raw')
        
    print("Cleaning up zip file...")
    os.remove(zip_path)
    print("Data successfully downloaded and extracted to data/raw!")

if __name__ == '__main__':
    download_and_extract()