import lightning as L
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .mlp_model_manager import ModelManager


class L_Net(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer, out_path, save_model=True):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.error_train = []
        self.error_test = []

        self.best_error_train = np.inf
        self.best_error_test = np.inf

        self.out_dict = {}

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.6, patience=10)

        self.model_manager = ModelManager(out_path=out_path, save_model=save_model)

    def configure_optimizers(self):
        return self.optimizer

    def network_step(self, batch):
        emb_2, emb_3, X, lab = batch
        logits = self.model(emb_2, emb_3, X)
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
        train_error = np.nanmean(self.error_train)
        print(f'[INFO] Train error: {train_error}')
        if train_error < self.best_error_train:
            self.best_error_train = train_error

        self.error_train = []

    def on_validation_epoch_end(self):
        test_error = np.nanmean(self.error_test)
        print(f'[INFO] Val error: {test_error}')
        if test_error < self.best_error_test:
            self.best_error_test = test_error

        self.scheduler.step(metrics=test_error)
        print(f'LR: {self.optimizer.param_groups[0]["lr"]}')

        self.model_manager.save_model(self.model, test_error)

        self.error_test = []

    def on_train_end(self):
        print(f'[INFO] END EPOCH\nTrain error: {self.best_error_train}\nVal error: {self.best_error_test}')
        self.out_dict['rmse_train'] = self.best_error_train
        self.out_dict['rmse_test'] = self.best_error_test


import lightning as L
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .mlp_model_manager import ModelManager


class L_Net_TL(L.LightningModule):
    def __init__(self, model, base_model, loss_fn, optimizer, out_path, save_model=True):
        super().__init__()
        self.model = model
        self.base_model = base_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.error_train = []
        self.error_test = []

        self.best_error_train = np.inf
        self.best_error_test = np.inf

        self.out_dict = {}

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)

        self.model_manager = ModelManager(out_path=out_path, save_model=save_model)

    def configure_optimizers(self):
        return self.optimizer

    def network_step(self, batch):
        emb_2, emb_3, X, lab = batch
        base_out = self.base_model(emb_2, emb_3, X)
        logits = self.model(base_out)
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
        train_error = np.nanmean(self.error_train)
        print(f'[INFO] Train error: {train_error}')
        if train_error < self.best_error_train:
            self.best_error_train = train_error

        self.error_train = []

    def on_validation_epoch_end(self):
        test_error = np.nanmean(self.error_test)
        print(f'[INFO] Val error: {test_error}')
        if test_error < self.best_error_test:
            self.best_error_test = test_error

        self.scheduler.step(metrics=test_error)
        print(f'LR: {self.optimizer.param_groups[0]["lr"]}')

        self.model_manager.save_model(self.model, test_error)

        self.error_test = []

    def on_train_end(self):
        print(f'[INFO] END EPOCH\nTrain error: {self.best_error_train}\nVal error: {self.best_error_test}')
        self.out_dict['rmse_train'] = self.best_error_train
        self.out_dict['rmse_test'] = self.best_error_test
