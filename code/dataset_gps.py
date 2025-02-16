import pickle

import pdb
import os
import re
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import json
import datetime



def load_gps():  
    with open('./data/gps/shanghai.txt', 'r') as f:
        data = []   
        for line in f:
            line_data_str = line.strip().split(' ')  
            line_data_int = [int(x) for x in line_data_str]  
            data.append(line_data_int)   
    return data

def gps_embedding(data, input_size=4096, embedding_dim=35):  
    embed_dict = np.load('./embedding/embedding_dict.npy', allow_pickle=True).item()

    data = np.array(data)

    embedded_data = np.zeros((data.shape[0], data.shape[1], embedding_dim))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            embedded_data[i][j] = embed_dict[data[i][j]]
        
    embedded_data = torch.tensor(embedded_data)
    
    return embedded_data

def load_flow():
    with open('./data/flow/flow.json','r') as f:
        date2flowmat = json.load(f)
    train_data = []
    for k,v in date2flowmat.items():
        train_data.append(v)
    train_data = np.array(train_data)
    M, m = np.max(train_data), np.min(train_data)
    train_data = (2 * train_data - m - M) / (M - m)

    return train_data.tolist(), m, M

    
class GPS_Dataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)  

        self.embedded_data = []
        self.observed_masks = []
        self.gt_masks = []
        path = (
            "./data/gps_missing" + str(missing_ratio) + "_seed" + str(seed) + ".pk"
        )

       
        data = load_gps()
        data = torch.LongTensor(data)
        embedded_data = gps_embedding(data)
        self.embedded_data = embedded_data.detach().numpy()
        self.observed_masks = torch.ones_like(embedded_data).detach().numpy()
        self.gt_masks = torch.zeros_like(embedded_data).detach().numpy()  
        flow_data = load_flow()
        self.flow_data = np.array(flow_data)
            
        with open(path, "wb") as f:
            pickle.dump(
                [self.embedded_data, self.observed_masks, self.gt_masks], f
            )
       
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.embedded_data))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "flow_data": self.flow_data[index],
            "observed_data": self.embedded_data[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.0):

    dataset = GPS_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))
    print(indlist)
    np.random.seed(seed)
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    dataset = GPS_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = GPS_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = GPS_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader





