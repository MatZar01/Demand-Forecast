# Demand-Forecast

## Initial info

In current form, repository is fit to run on [Demand-Forecasting](https://www.kaggle.com/datasets/aswathrao/demand-forecasting) 
Kaggle dataset, and it will be suited for other, more general datasets in the future.
No argv manager is available at this point, so each directory string has to be updated manually within the edited file.

Below are the instructions for running the experiments on Demand-Forecasting dataset.

---

## Running ARIMA and DTs

Both ARIMA and DTs are suited only for **Store-SKU** pair training.

### ARIMA

ARIMA predicts next time series value only on the basis of the previous ones. Hence, no features are actually used
in the training, and the data has to be divided into **Store-SKU** pairs - as rows in the dataset do not make up a 
continuous time series for the whole dataset. ARIMA is executed by `/baseline/baseline_ARIMA.py`.

### DTs

Decision Trees and Random Forests can be trained using `/baseline/baseline_DT.py` script. Unlike ARIMA, they take into 
account all the features in the dataset. For experiments, `lag`, `max depth`, `leaf nodes` and `number of estimators` 
can be tuned. 

---

## Running Baseline MLP experiments

All the baseline **MLP** experiments use built-in embedding layers, so initial embedders training is not required 
at this point.

### Obtaining one-hot labels from the dataset

To get one-hot labels, run `make_onehots.py` from `/baseline` directory. It runs _Dataloader_ form `name_embedding.py` 
that is obsolete now, and there is no need to use it anymore. Obtained one-hot embedders are valid for all the rest of
scripts in the repository.

- `DATA_PATH` should point to the dataset training `.csv` file.
- `OUT_PATH` is where `onehot_C2.pkl` and `onehot_C3.pkl` will be saved.

### Running base MLP 

Baseline MLP experiments are run with `baseline_MLP_emb.py` from `/baseline` directory. 

In `embedders` dict, you have to provide paths for .pkl files with previously obtained onehot-embeddings.

Notable Parameters:

- `QUANT` set `True` to include past sales in the feature vector, `False` to exclude it
- `NORMALIZE` set `True` to normalize feature vector
- `MATCHES_ONLY` **IMPORTANT** set `True` to train series of models for each **Store-SKU** pair. If set to `False` 
**ONE MODEL WILL BE TRAINED FOR ENTIRE DATASET**
- `OUT_PATH` path where trained model will be saved (for use with `MATCHES_ONLY` set to False) along with the result dictionary
- `SAVE_MODEL` set `True` to save model in `OUT_PATH`

Two base models (found in `baseline/base_src/mlp_models.py`) can be trained with this script:

1. `MLP_emb` for later finetuning with `mlp_finetuning.py`
2. `MLP_emb_tl` for later transfer learning with `mlp_transfer.py`

**NOTE:** `baseline_MLP_emb.py` has two major functions:

1. To train series of models for each **Store-SKU** pair - acting as a baseline for the best scenario comparison where 
every model covers only one pair. Those models **WILL NOT** be finetuned or transfer learned, so there is no need for 
saving them locally.
2. **To pre-train and save a single model that can later be finetuned** (or transfer learned in case of TL model used) -
this should be the only case where you'd like to save the model for later.

### Running MLP finetuning (BP MLP)

Finetuning is performed with `mlp_finetuning.py`.

By setting up `MODEL_PATH`, model can be finetuned on each **Store-SKU** pair when setting up `MATCHES_ONLY` to `True` 
or again on the whole dataset (_e.g._ with different loss or optimizer) by setting it to `False`.

### Running MLP transfer learning

TL is performed with `mlp_transfer.py`.

Similarly to finetuning, `MODEL_PATH` pointing to `MLP_emb_tl` model trained with `baseline_MLP_emb.py` is required, 
along with setting up `MATCHES_ONLY` to `True` in order to TL on **Store-SKU** pairs. Note, that this time only last 
regression layer is trained.

---

## Additional MLP experiments

### Running MLP clustered models

#### Embedding clustering

You can perform embedding clustering with `embedding_clustering_data.py`.

The only requirement is to have a general `MLP_emb` (or `_tl`) model trained and saved with `MATCHES_ONLY` set to `False`.

In order to run script provide:

- `MODEL_PATH` where the saved model is stored
- `OUT_DICT_PATH` where `.pkl` file of `out_dict` will be stored
- `ASSIGNMENT_PATH` where `.pkl `file of clustering assignments will be stored

#### Running model

Update `MATCHES_PATH` to be pointing at `emb_assignments` `.pkl` file and run the script to obtain coarse-clustered models
for the whole dataset. Note that groups in `out_dict`, `model_groups` and last `print` should match the groups used for 
clustering in previous step.

### Running grouped MLP models

Grouped MLPs are trained using `baseline_MLP_gruper.py`. This function is in experimental phase and will be updated as
the project progresses.

---

## Additional info

Models are trained with **_Reduce On Plateau_** scheduler - if you'd like to update its parameters, change it directly in
`baseline/base_src/mlp_trainer.py`, lines 25/104 (for with/without TL).

Updates for the exchangeable feature-length datasets will be added as the project progresses.