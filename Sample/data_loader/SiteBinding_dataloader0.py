import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd
import csv


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 注意表头
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            # print(counter)
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df):
        self.data = df
        self.df_length = len(df[0])
        self.x_idx = self.get_idx()

    def __getitem__(self, index):
        hi = self.x_idx[index]
        # print(hi)
        lo = hi - 1
        Features = self.data[0][lo: hi][0]
        Onehot = self.data[1][lo: hi][0][:-1]
        Labels = self.data[1][lo: hi][0]
        # Labels = self.data[2][lo: hi][0]

        # 结合位点、时序训练数据
        features, Onehots, labels = self.get_data(Features, Onehot, Labels)
        features = torch.from_numpy(features).type(torch.float)
        labels = torch.from_numpy(labels).type(torch.float)
        Onehots = torch.from_numpy(Onehots).type(torch.float)

        return features, Onehots, labels  # train+target

    def __len__(self):
        return len(self.x_idx)

    def get_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(1, self.df_length)
        x_end_idx = [x_index_set[j] for j in range((len(x_index_set)))]
        return x_end_idx

    def get_data(self, Features, Onehot, Labels):
        # 包含四类数据，其中onehotlabel中只需要其中的标签数据

        Featurex = Features
        Onehot = Onehot
        labels = Labels[-1]

        # temp=onehot_labelx[:-1]
        # features=[]
        # i=0
        # while i<len(temp)-1:
        #     features.append(temp[i:i+4])
        #     i+=4

        return np.array(Featurex, dtype='float64'), np.array(Onehot), np.array(labels)


# 数据读取测试
if __name__ == '__main__':
    print("done!")
