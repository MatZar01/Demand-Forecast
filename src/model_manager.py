import numpy as np
import torch
import copy


class ModelManager:
    def __init__(self, out_path, col):
        self.out_path = out_path
        self.last_low_error = np.inf
        self.col = col

    def save_model(self, model, error):
        if error < self.last_low_error:
            model = copy.deepcopy(model)
            embedder = copy.deepcopy(model.embedder)
            self.last_low_error = error

            torch.save(model.cpu(), f'{self.out_path}/model_c{self.col}.pth')
            torch.save(embedder.cpu(), f'{self.out_path}/embedder_c{self.col}.pth')
