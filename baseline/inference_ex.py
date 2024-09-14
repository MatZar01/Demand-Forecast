from base_src import MLP_dataset_emb
from base_src import MLP, MLP_emb, MLP_emb_tl
import torch
from torch.utils.data import DataLoader

"""
Hi, in this simple file I'll walk you through the inference with the models
models should be in ../inference_tests_models as .pth files 
If you'd have questions, feel free to ask me on discord
"""

DEVICE = 'cuda'  # can be set to 'cpu', in this script it won't matter anyhow
BATCH = 1  # batch is set to 1 just for convenience
LAG = 15  # how many samples are in the series -- it depends on the model architecture, so 15 it is
QUANT = True  # if to add previous sales to input vector
EMBED = True  # if to use embedding (onehot embedding to int for cat2vec embedder)
NORMALIZE = True  # if to normalize the rest of input vector
MATCHES_ONLY = False  # if to only select single SKU-Store match from dataloader - affects m = None

if not MATCHES_ONLY:
    m = None

# select train .csv from dataset (in test .csv there are no labels)
DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/demand-forecasting/train.csv'

# embedders are only used for onehot-to-int encoding for cat2vec -- they are saved to omit confusion between models
embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}


# load the model - there are two: mlp_model_for_ft was for finetuning, mlp_model_for_tl was for transfer learning
# they both work the same, only difference is that mlp_model_for_tl has exchangeable classifier layer
model = torch.load('/home/mateusz/Desktop/Demand-Forecast/inference_tests_models/mlp_model_for_tl.pth')


# load dataloader with previous parameters
val_data = MLP_dataset_emb(path=DATA_PATH, train=False, lag=LAG, get_quant=QUANT, normalize=NORMALIZE,
                           embedders=embedders, matches=m)

# just an ordinary torch DataLoader - update num_workers to number ov available cores
val_dataloader = DataLoader(val_data, batch_size=BATCH, shuffle=True, num_workers=15)

# get single batch of data
emb_col_2, emb_col_3, feature_vec, y = next(iter(val_dataloader))
# when batch = 1:
# it should be: column 2 onehot: LongTensor (size 1xlag),
# column 3 onehot: LongTensor (size 1xlag),
# feature vector: Tensor (size 1xlagx5),
# label: Tensor (size 1x1)

# get output from model
model_output = model(emb_col_2, emb_col_3, feature_vec)
# it should be: Tensor (size 1x1)


# Now if you'd like to just use embedding layers (embedder_2 and _3 works the same)
# get embedding layer
embedder_col_2 = model.embedder_2

# use it on sample from dataloader
# here it's first element of the first batch
embedding_out = embedder_col_2(emb_col_2[0][0])
# it should give you categorical embedding (Tensor size 5,)
