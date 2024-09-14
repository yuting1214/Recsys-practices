import os
import io
import zipfile
import requests
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split

class BaseDataset(Dataset):
    def split(self, *args, **kwargs):
        """Split the dataset into train and test datasets."""
        raise NotImplementedError

class ML100K(BaseDataset):
    def __init__(self, data_dir="./data/movielens", normalize_rating=False, load_ratio=1.0):
        """
        Initialize the ML100K dataset.
        
        Parameters:
        - data_dir: Directory where the data is stored.
        - normalize_rating: If True, normalize ratings to be between 0 and 1.
        - load_ratio: A float between 0 and 1 to control the percentage of data to load.
        """
        self.normalize_rating = normalize_rating
        self.data_dir = data_dir
        self.df = self.read_data_ml100k(data_dir)
        self.load_ratio = load_ratio

        # Shuffle and reduce dataset if load_ratio is less than 1.0
        if self.load_ratio < 1.0:
            self.df = self.sample_data(self.df, self.load_ratio)

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

    def sample_data(self, df, load_ratio):
        """ 
        Sample a percentage of the data.
        
        Parameters:
        - df: Original DataFrame
        - load_ratio: Percentage of data to load (0 < load_ratio <= 1.0)
        
        Returns:
        - A new DataFrame sampled according to load_ratio.
        """
        sample_size = int(len(df) * load_ratio)
        return df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    def read_data_ml100k(self, data_dir="./data/movielens") -> pd.DataFrame:
        """
        Reads the ML-100K data from the specified directory.
        
        Parameters:
        - data_dir: Directory to load the data from.
        
        Returns:
        - DataFrame containing the dataset.
        """
        data_file = os.path.join(data_dir, 'ml-100k/u.data')
        if not os.path.exists(data_file):
            self.download_and_extract(data_dir)
        
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        return pd.read_csv(data_file, sep='\t', names=names)

    def download_and_extract(self, data_dir):
        """
        Downloads and extracts the ML-100K dataset if not already available.
        
        Parameters:
        - data_dir: Directory to store the dataset.
        """
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        file_path = os.path.join(data_dir, 'ml-100k')
        if not os.path.exists(file_path):
            print(f"Downloading {url}...")
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                print(f"Extracting to {data_dir}...")
                z.extractall(data_dir)
        else:
            print("Dataset already exists, skipping download.")

    def split(self, train_ratio=0.8):
        """
        Splits the dataset into training and testing sets.
        
        Parameters:
        - train_ratio: The ratio of the dataset to use for training.
        
        Returns:
        - Two random_split objects (train and test datasets).
        """
        train_len = int(train_ratio * len(self))
        test_len = len(self) - train_len
        return random_split(self, [train_len, test_len])

    def __len__(self):
        """
        Returns the total length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the user_id, item_id, and rating for a given index.
        
        Parameters:
        - idx: Index of the data point.
        
        Returns:
        - user_id: Tensor containing the user ID.
        - item_id: Tensor containing the item ID.
        - rating: Tensor containing the rating.
        """
        user_id = torch.tensor(self.user_id[idx], dtype=torch.long)
        item_id = torch.tensor(self.item_id[idx], dtype=torch.long)
        rating = torch.tensor(self.rating[idx], dtype=torch.float32)
        
        return user_id, item_id, rating