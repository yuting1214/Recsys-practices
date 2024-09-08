import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split

class BaseDataset(Dataset):
    def split(self, *args, **kwargs):
        """Split the dataset into train and test datasets."""
        raise NotImplementedError

class ML100K(BaseDataset):
    def __init__(self, data_dir="./data/movielens/ml-100k", normalize_rating=False):
        self.normalize_rating = normalize_rating
        self.data_dir = data_dir
        self.df = self.read_data_ml100k(data_dir)
        self.df.user_id -= 1
        self.df.item_id -= 1
        if normalize_rating:
            self.df.rating /= 5.0
        self.num_users = self.df.user_id.unique().shape[0]
        self.num_items = self.df.item_id.unique().shape[0]
        self.user_id = self.df.user_id.values
        self.item_id = self.df.item_id.values
        self.rating = self.df.rating.values.astype(np.float32)
        self.timestamp = self.df.timestamp

    def read_data_ml100k(self, data_dir="./data/movielens/ml-100k") -> pd.DataFrame:
        data_file = os.path.join(data_dir, 'u.data')
        if not os.path.exists(data_file):
            self.download_and_extract(data_dir)
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        return pd.read_csv(data_file, sep='\t', names=names)

    def download_and_extract(self, data_dir):
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        if not os.path.exists(os.path.join(data_dir, 'ml-100k.zip')):
            print(f"Downloading {url}...")
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                print(f"Extracting to {data_dir}...")
                z.extractall(data_dir)
        else:
            print("Dataset already exists, skipping download.")

    def split(self, train_ratio=0.8):
        train_len = int(train_ratio * len(self))
        test_len = len(self) - train_len
        return random_split(self, [train_len, test_len])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]
