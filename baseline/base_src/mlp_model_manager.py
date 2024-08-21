import numpy as np
import torch
import copy


class ModelManager:
    def __init__(self, out_path):
        self.out_path = out_path
        self.last_low_error = np.inf

    def save_model(self, model, error):
        if error < self.last_low_error:
            model = copy.deepcopy(model)
            self.last_low_error = error

            torch.save(model.cpu(), f'{self.out_path}/mlp_model.pth')
