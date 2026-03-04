import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd
import csv

import math
import random
import csv
import torch
import numpy as np

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
import re


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
        self.target_length = 8

    def __getitem__(self, index):
        # print(len(self.data))
        hi = self.x_idx[index]
        # print(hi)
        lo = hi - 1
        graph = self.data[0][lo: hi][0]  # 二级结构图
        Loop_graph = self.data[1][lo: hi][0]  # 二级结构图
        onelabels = self.data[2][lo: hi][0]  # 16维的包含标签和onehot等理化性质的
        oneFeature = self.data[3][lo: hi][0]  # ELMo的标签
        AllRegionIndex = self.data[4][lo: hi][0]  # ELMo的标签

        # 结合位点、时序训练数据
        graph, Loop_graph, labels, onehots, onehot_feature, RegionIndex = self.get_data(graph, Loop_graph, onelabels,oneFeature, AllRegionIndex)  # 分别为结构图，序列标签，标签图，标签图节点初始特征 (onehot+理化性质)，结构图节点初始特征
        graph = torch.from_numpy(graph).type(torch.float)
        Loop_graph = torch.from_numpy(Loop_graph).type(torch.float)
        labels = torch.from_numpy(labels).type(torch.float)
        onehot_feature = torch.from_numpy(onehot_feature).type(torch.float)

        return graph, Loop_graph, labels, onehots, onehot_feature, RegionIndex  # train+target

    def match_motif(self, sequence, motifs):
        for motif in motifs:
            if re.search(motif, sequence):
                return True
        return False

    def detect_secondary_structure(self, structure_matrix):
        if not isinstance(structure_matrix, list):
            raise ValueError("Input structure_matrix must be a list.")

        structure_matrix_np = np.array(structure_matrix)
        has_nonzero = np.any(structure_matrix_np != 0)

        return has_nonzero

    def identify_key_regions(self, A, B, motifs):
        sequence = ''.join(['A' if A[j] == 1.0 else
                            'C' if A[j] == 2.0 else
                            'G' if A[j] == 3.0 else
                            'U' if A[j] == 4.0 else 'N' for j in range(len(A))])
        structure_matrix = B

        max_score = float('-inf')
        best_start, best_end = 0, 0
        found_match = False

        for start in range(len(A) - self.target_length + 1):
            for end in range(start + 1, min(start + self.target_length + 1, len(A))):
                sub_seq = sequence[start:end]
                sub_matrix = structure_matrix[start:end]

                motif_score = int(self.match_motif(sub_seq, motifs))
                structure_score = int(self.detect_secondary_structure(sub_matrix))
                score = motif_score + structure_score

                if self.match_motif(sub_seq, motifs) and self.detect_secondary_structure(sub_matrix):
                    score += 100
                elif self.match_motif(sub_seq, motifs):
                    score += 50
                elif self.detect_secondary_structure(sub_matrix):
                    score += 20

                if score > max_score:
                    max_score = score
                    best_start, best_end = start, end - 1
                    found_match = True

        if not found_match:
            best_start = random.randint(0, len(A) - self.target_length)
            best_end = best_start + self.target_length - 1

        return [best_start, best_end]

    def __len__(self):
        return len(self.x_idx)

    def get_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(1, self.df_length)
        x_end_idx = [x_index_set[j] for j in range((len(x_index_set)))]
        return x_end_idx

    def get_data(self, graph, Loop_graph, onelabels, oneFeature, AllRegionIndex):
        # 包含四类数据，其中onehotlabel中只需要其中的标签数据
        graphx = graph
        Loop_graphx = Loop_graph

        onehot_labelx = onelabels
        onehot_featurex = oneFeature

        labels = onehot_labelx[-1]
        onehots = onehot_labelx[:-1]  # 类型list

        RegionIndex = AllRegionIndex

        # temp=onehot_labelx[:-1]
        # features=[]
        # i=0
        # while i<len(temp)-1:
        #     features.append(temp[i:i+4])
        #     i+=4

        return np.array(graphx, dtype='float64'), np.array(Loop_graphx, dtype='float64'), np.array(labels), np.array(
            onehots, dtype='float64'), np.array(onehot_featurex, dtype='float64'), np.array(RegionIndex, dtype='int')


# 数据读取测试
if __name__ == '__main__':
    print("done!")
