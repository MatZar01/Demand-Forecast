from base_src import MLP_dataset_emb
from base_src import MLP_emb, MLP_emb_tl
import torch
from torch.utils.data import DataLoader
from base_src import get_matches
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

DEVICE = 'cuda'

DATA_PATH = '/home/mateusz/Desktop/Demand-Forecast/DS/productdemandforecasting/train.csv'
matches = get_matches(DATA_PATH, encode=True)

MODEL_PATH = '/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer/mlp_model_3.pth'

OUT_DICT_PATH = f'/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer/emb_data_3.pkl'
ASSIGNMENT_PATH = f'/home/mateusz/Desktop/Demand-Forecast/baseline/results_mlp/transfer/emb_assignments_3.pkl'

embedders = {'C2': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C2.pkl'},
             'C3': {'onehot': '/home/mateusz/Desktop/Demand-Forecast/embedding_models/onehot_C3.pkl'}}


model = torch.load(MODEL_PATH)

onehot_2 = pickle.load(open(embedders['C2']['onehot'], 'rb'))
onehot_3 = pickle.load(open(embedders['C3']['onehot'], 'rb'))

embedder_2 = model.embedder_2
embedder_3 = model.embedder_3

out_dict = {}
data = []

for i in tqdm(matches):
    col_2 = i[0]
    col_3 = i[1]

    #col_2_onehot = np.argmax(onehot_2.transform(np.array(col_2).reshape(1, -1).astype(np.object_)).toarray())
    #col_3_onehot = np.argmax(onehot_3.transform(np.array(col_3).reshape(1, -1).astype(np.object_)).toarray())

    col_2_emb = embedder_2(torch.LongTensor([col_2]))
    col_3_emb = embedder_3(torch.LongTensor([col_3]))

    concat = torch.concatenate([col_2_emb, col_3_emb]).flatten().detach().numpy()
    c2 = col_2_emb.flatten().detach().numpy()
    c3 = col_3_emb.flatten().detach().numpy()

    out_dict[f'{col_2}_{col_3}'] = {'concat': concat, '2': c2, '3': c3}
    data.append(concat)

data = np.array(data)

inertia = []
cluster_num = []
for k in tqdm(range(1, 50)):
    clustering = KMeans(n_clusters=k, random_state=1, init='k-means++', max_iter=500, n_init='auto')
    clustering.fit(data)
    inertia.append(clustering.inertia_)
    cluster_num.append(k)

plt.plot(cluster_num, inertia)
plt.grid()
plt.title('Sludge Curve')
plt.xlabel('clusters')
plt.ylabel('inertia')
plt.show()

clusters = [10, 15, 20]  # picked on the basis of the sludge curve

for c in clusters:
    clustering = KMeans(n_clusters=c, random_state=1, init='k-means++', max_iter=500, n_init='auto')
    clustering.fit(data)
    assignments = clustering.labels_

    for (i, j) in enumerate(out_dict.keys()):
        out_dict[j][c] = assignments[i]

# let's group them all
assignments_dict = {10: {}, 15: {}, 20: {}}
for c in clusters:
    for x in range(c):
        assignments_dict[c][x] = []

for key in out_dict.keys():
    assignments_dict[10][out_dict[key][10]].append(key)
    assignments_dict[15][out_dict[key][15]].append(key)
    assignments_dict[20][out_dict[key][20]].append(key)

pickle.dump(out_dict, open(OUT_DICT_PATH, 'wb'))
pickle.dump(assignments_dict, open(ASSIGNMENT_PATH, 'wb'))

