import torch
import lightning as L
from torchmetrics import MeanSquaredError

class LitModel(L.LightningModule):
    """Template Lightning Module to train model"""

    def __init__(self, model_class, lr=0.002, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model_class(**kwargs)
        self.lr = lr
        self.sparse = getattr(self.model, "sparse", False)
        self.train_metric = None
        self.valid_metric = None

    def configure_optimizers(self):
        if self.sparse:
            return torch.optim.SparseAdam(self.parameters(), self.lr)
        else:
            return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def get_label(self, batch):
        raise NotImplementedError()

    def forward(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        # Extract the features and labels from the batch
        y = self.get_label(batch)
        preds = self(batch)
        
        # Print debug info for preds and y
        print(f"Batch Index: {batch_idx}")
        print(f"Predictions (preds): {preds}")
        print(f"Ground Truth (y): {y}")
        
        # Add a debug print to check the types and shapes
        print(f"Preds type: {type(preds)}, shape: {preds.shape}")
        print(f"Labels type: {type(y)}, shape: {y.shape}")

        # Compute the loss using the train_metric function (likely a loss function)
        batch_loss = self.train_metric(preds, y)
        
        # Print the calculated loss
        print(f"Batch Loss: {batch_loss}")

        # Log the loss for this step
        self.log("train_loss_step", batch_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return batch_loss


    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.train_metric.compute())
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        y = self.get_label(batch)
        preds = self(batch)
        self.valid_metric.update(preds, y)

    def on_validation_epoch_end(self):
        self.log('valid_loss_epoch', self.valid_metric.compute())
        self.valid_metric.reset()

class LitMF(LitModel):
    """A specific implementation for Matrix Factorization tasks."""

    def __init__(self, model_class, lr=0.002, **kwargs):
        super().__init__(model_class, lr, **kwargs)

        # Declare specific metrics (train_metric, valid_metric)
        self.train_metric = MeanSquaredError(squared=False)  # RMSE for training
        self.valid_metric = MeanSquaredError(squared=False)  # RMSE for validation

    def get_label(self, batch):
        """Extract ground truth labels from the batch."""
        return batch[-1]  # Assuming ratings are the last element of the batch

    def forward(self, batch):
        """Forward pass through the MF model."""
        user_ids, item_ids, _ = batch
        return self.model(user_ids, item_ids)