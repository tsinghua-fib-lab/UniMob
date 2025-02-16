import torch
import pickle 
import pdb
from annoy import AnnoyIndex

import numpy as np

def postprocess(query_vector):
    n_trees = 10  
    embed_dict = np.load('embedding_dict.npy', allow_pickle=True).item()

    n_features = 35
    annoy_index = AnnoyIndex(n_features, metric='angular')


    key_list = []
    value_list = []
    t = 0
    for key, value in embed_dict.items():
        # print(key)
        key_list.append(key)
        value_list.append(value)
        annoy_index.add_item(t, value)
        t = t + 1

    annoy_index.build(n_trees)

    # query_vector = np.load('./embedding/0sample.npy', allow_pickle=True)
    query_vector = torch.tensor(query_vector)
    query_vector = query_vector.permute(1, 0, 2, 3)
    # pdb.set_trace()
    # query_vector = [random.gauss(0, 1) for _ in range(n_features)]
    n_neighbors = 1
    vec_list = []
    for i in range(query_vector.shape[1]):
        for j in range(query_vector.shape[2]):
            nearest_neighbors = annoy_index.get_nns_by_vector(query_vector[0][i][j], n_neighbors)
            # pdb.set_trace()
            vec_list.append( key_list[nearest_neighbors[0]] )
    
    vec_list = np.array(vec_list)
    vec_list = vec_list.reshape((-1, 48))
    vec_list = torch.tensor(vec_list)
    # pdb.set_trace()
    print(vec_list.size())
    return vec_list

# with open("predicted_outputs_nsample1.pk", "rb") as f:
#     data = pickle.load(
#         f
# )
# # print(data)
# embedding_data = data[0]
# # embedding_data = torch.tensor(embedding_data)
# # pdb.set_trace()
# vec_list = postprocess(query_vector = embedding_data)
# np.save("sample_ip.npy", vec_list)
# # pdb.set_trace()

data = np.load("sample_ip.npy", allow_pickle=True)
np.savetxt("prediction.txt", data, delimiter=" ", newline="\n", fmt="%.0f")
pdb.set_trace()
