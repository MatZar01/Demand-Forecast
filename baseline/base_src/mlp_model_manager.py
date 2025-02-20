import numpy as np
import torch
import copy


class ModelManager:
    def __init__(self, out_path, save_model: bool = True):
        self.out_path = out_path
        self.last_low_error = np.inf
        self.save = save_model

    def save_model(self, model, error):
        if error < self.last_low_error:
            self.last_low_error = error
            if self.save:
                model = copy.deepcopy(model)
                torch.save(model.cpu(), f'{self.out_path}/mlp_model_3.pth')
