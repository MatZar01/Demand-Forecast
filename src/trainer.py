import lightning as L
import numpy as np
from .model_manager import ModelManager


class L_Net(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer, out_path, col):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.error_train = []
        self.error_test = []

        self.model_manager = ModelManager(out_path, col)

    def configure_optimizers(self):
        return self.optimizer

    def network_step(self, batch):
        X, lab = batch
        logits = self.model(X)
        loss = self.loss_fn(logits, lab)
        return logits, loss

    def training_step(self, batch):
        logits, loss = self.network_step(batch)
        self.error_train.append(loss.detach().cpu().numpy())
        return loss

    def validation_step(self, batch):
        logits, loss = self.network_step(batch)
        self.error_test.append(loss.detach().cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        print(f'[INFO] Train error: {np.mean(self.error_train)}')
        self.error_train = []

    def on_validation_epoch_end(self):
        print(f'[INFO] Val error: {np.mean(self.error_test)}')
        self.model_manager.save_model(self.model, np.mean(self.error_test))
        self.error_test = []

    def on_train_end(self):
        print(f'[INFO] END EPOCH\nTrain error: {self.error_train}\nVal error: {self.error_test}')


