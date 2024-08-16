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
            embedder_2 = copy.deepcopy(model.embedder_2)
            embedder_3 = copy.deepcopy(model.embedder_3)
            self.last_low_error = error

            torch.save(model.cpu(), f'{self.out_path}/model.pth')
            torch.save(embedder_2.cpu(), f'{self.out_path}/embedder_c2.pth')
            torch.save(embedder_3.cpu(), f'{self.out_path}/embedder_c3.pth')
