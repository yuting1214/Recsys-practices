import lightning as L
from torch.utils.data import DataLoader
from src.data_loader.dataset import BaseDataset

class LitDataModule(L.LightningDataModule):
    def __init__(self, dataset: BaseDataset, train_ratio: float = 0.8, batch_size: int = 32, num_workers: int = 2):
        super().__init__()
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.val_dataset = self.dataset.split(self.train_ratio)
        elif stage == "test":
            self.test_dataset = self.dataset.split(self.train_ratio)[1]
        elif stage == "predict":
            self.predict_dataset = self.dataset.split(self.train_ratio)[1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
