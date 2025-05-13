
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from torch.autograd import Variable
from .Utils import *
import pandas as pd
import csv
from .configure import *
from .ConsensusComponents import *

from sklearn.preprocessing import normalize
import numpy as np
from scipy.special import iv
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from .Graph_Wavelet import *
from scipy.integrate import quad
import time
from .SemanticNet import *


def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
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


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# class 1: 拉普拉斯矩阵生成（池化+（D-W））+可逆
# class 2: 空间对齐（切比雪夫近似） +可逆
# class 3: 孪生-耦合 +可逆

class PoolingLayer(nn.Module):
    # def __init__(self, size, batch, device="cuda:0"):
    def __init__(self, size, device="cuda:0"):
        super(PoolingLayer, self).__init__()
        self.size = size
        # self.batch = batch
        # self.weight = nn.Parameter(torch.rand(size=(self.batch, self.size, self.size)))
        self.weight = nn.Parameter(torch.rand(size=(1, self.size, self.size)))
        self.to(device)

        return

    def forward(self, A):
        A = torch.mul(self.weight, A)
        A = 0.5 * (A + A.permute(0, 2, 1))  # 可逆部分需要写出这部分的计算
        # print(self.weight)
        return A


class TransposeHook:  # 保证在更新梯度时，U和L的权重互为转置，这样的操作可能会影响模型的训练和收敛性
    def __init__(self, source_layer):
        self.source_layer = source_layer

    def __call__(self, grad):
        self.source_layer.weight.data.copy_(self.source_layer.weight.data.permute(0, 2, 1))
        return grad


"""
共识基序网络需要完成的内容：
1. 潜在关键子序列的筛选
2. motif graph 构建（不考虑二级结构）
3. consensus graph 构建
4. 关键区域和非关键区域的对比+特征的融合
5. 图卷积
6. Prediction
"""


class ScalingLayer(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(41, data_dim, requires_grad=True))

    def forward(self, x):  # x为【b,41,256】
        scaling_coff = torch.exp(self.log_scale_vector)
        # print(type(scaling_coff))
        x = scaling_coff.unsqueeze(0) * x
        return x, scaling_coff


class ConsensusMotifNetwork(nn.Module):  # 当前为一层的卷积过程,默认均为三维张量
    def __init__(self, num, size, batch, batch1, device="cuda:0"):
        super(ConsensusMotifNetwork, self).__init__()

        self.size = size
        self.batch = batch
        self.batch1 = batch1
        self.num = num
        self.device = device

        self.bn = nn.BatchNorm1d(41)  # 表征的标准化
        self.bnSR = nn.BatchNorm1d(15)  # 表征的标准化
        self.bn1 = nn.BatchNorm1d(64)  # 表征的标准化
        self.scalingLayer = ScalingLayer(64)

        self.graph_wavelet_layer = GraphWaveletNeuralNetwork(feature_dims=64, hidden_dims=64, device=self.device,
                                                             dropout_rate=0.2)

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        self.shape1 = nn.Sequential(
            nn.Linear(64, 64),
            # nn.Tanh(),
            nn.ReLU(),  # 权重
            nn.Linear(64, 64),
        )

        self.scalingWeight = nn.Sequential(
            nn.Linear(41, 41),
            # nn.Softmax(),
            nn.Sigmoid(),
            # nn.PReLU(),  # 权重
        )

        self.scalingWeight_SR = nn.Sequential(
            nn.Linear(15, 15),
            # nn.Softmax(),
            nn.Sigmoid(),
            # nn.PReLU(),  # 权重
        )

        self.prob = nn.Sequential(
            nn.Linear(64, 1),
            # nn.Softmax(),  # 权重
            nn.Sigmoid(),  # 权重
        )

        self.gru = nn.GRU(271, 271)

        self.semantics = Semantic_network()

        self.RNASequenceFilter = RNASequenceFilter(device=self.device)  # 获得motif graph和最终的节点特征
        self.NodeConsensusVector = NodeConsensusVector(num_anchor_points=100,device=self.device)  # 获得共识图

        self.to(device)

    def laplacian_multi(self, W):  # 输入的是三维张量

        # hard_attention = max_min(W)
        hard_attention = W
        degree = torch.sum(hard_attention, dim=-1)
        hard_attention = 0.5 * (hard_attention + hard_attention.permute(0, 2, 1))  # 局部矩阵的
        degree_l = tensor_diag(degree)  # 度的对角矩阵 32x12x512x512
        diagonal_degree_hat = tensor_diag(1 / (torch.sqrt(degree) + 1e-6))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - hard_attention, diagonal_degree_hat))
        return laplacian

    def fourier(self, L, algo='eigh', k=100):
        """Return the Fourier basis, i.e. the EVD of the Laplacian."""

        L_array = L.cpu().detach().numpy()

        def sort(lamb, U):
            idx = lamb.argsort()
            return lamb[idx], U[:, idx]

        all_lamb = np.array([])
        all_U = np.array([])

        if algo is 'eig':
            for i in range(L_array.shape[0]):
                lamb, U = np.linalg.eig(L_array[i])
                lamb, U = sort(lamb, U)
                np.append(all_lamb, lamb)
                np.append(all_U, U)
        elif algo is 'eigh':
            temp_list_lamb = []
            temp_list_U = []
            for i in range(L_array.shape[0]):
                lamb, U = np.linalg.eigh(L_array[i])
                lamb, U = sort(lamb, U)
                temp_list_lamb.append(lamb)
                temp_list_U.append(U)
                all_lamb = np.stack(temp_list_lamb, axis=0)
                all_U = np.stack(temp_list_U, axis=0)
        elif algo is 'eigs':
            lamb, U = sp.linalg.eigs(L, k=k, which='SM')
            lamb, U = sort(lamb, U)
        elif algo is 'eigsh':
            lamb, U = sp.linalg.eigsh(L, k=k, which='SM')

        return all_lamb, all_U

    def largest_lamb(self, L, k=1, default_value=1.0):
        """Return the largest eigenvalue of the Laplacian for each graph in the batch."""

        # Ensure symmetry
        L = (L + L.permute(0, 2, 1)) / 2.0

        # Convert to NumPy array
        L_array = L.cpu().detach().numpy()

        epsilon = 1e-5  # Regularization parameter
        L_array += epsilon * np.eye(L_array.shape[1])

        temp_list_lamb = []
        for i in range(L_array.shape[0]):
            try:
                # Try ARPACK based method
                lamb, _ = sp.linalg.eigsh(L_array[i], k=k, which='LM', maxiter=10000)
            except (sp.linalg.ArpackNoConvergence, np.linalg.LinAlgError):
                # Fallback to numpy.linalg.eig
                try:
                    lamb, _ = np.linalg.eig(L_array[i])
                    lamb = np.real(lamb)  # Ensure real values
                    lamb = np.sort(lamb)[-k:]  # Get the largest k eigenvalues
                except np.linalg.LinAlgError:
                    lamb = np.array([default_value])

            temp_list_lamb.append(lamb[0])

        all_lamb = np.array(temp_list_lamb)

        return all_lamb

    import scipy.sparse.linalg as sp

    # def largest_lamb(self, L, k=1, maxiter=10000):
    #     """Return the Fourier basis, i.e. the EVD of the Laplacian."""
    #     L_array = L.cpu().detach().numpy()
    #
    #     temp_list_lamb = []
    #     for i in range(L_array.shape[0]):
    #         try:
    #             lamb, U = sp.eigsh(L_array[i], k=k, which='LM', maxiter=maxiter)
    #         except Exception as e:
    #             print(
    #                 f"Warning: No convergence for matrix {i} after {maxiter} iterations. Exception: {e}. Trying with higher iterations.")
    #             try:
    #                 lamb, U = sp.eigsh(L_array[i], k=k, which='LM', maxiter=maxiter * 2)
    #             except Exception as e:
    #                 print(f"Error: Still no convergence for matrix {i}. Exception: {e}. Skipping this matrix.")
    #                 continue  # 跳过这个矩阵
    #         temp_list_lamb.append(lamb[0])
    #     all_lamb = np.array(temp_list_lamb)
    #
    #     return all_lamb

    from scipy.sparse.linalg import svds

    # def largest_lamb(self, L, k=1):
    #     """Compute largest eigenvalues using SVD."""
    #     L_array = L.cpu().detach().numpy()
    #     temp_list_lamb = []
    #     for i in range(L_array.shape[0]):
    #         try:
    #             u, s, vt = svds(L_array[i], k=k)
    #             s = np.sort(s)[::-1]
    #             lamb = s[:k]
    #         except Exception as e:
    #             print(f"Warning: No convergence for matrix {i}. Exception: {e}. Skipping this matrix.")
    #             continue
    #         temp_list_lamb.append(lamb[0])
    #     all_lamb = np.array(temp_list_lamb)
    #     print(all_lamb.shape)
    #
    #     return all_lamb

    def weight_wavelet(self, s, lamb, U):

        s = s
        for i in range(len(lamb)):
            for j in range(len(lamb[0])):
                lamb[i][j] = math.exp(-lamb[i][j] * s)

        lamb = torch.exp(-lamb * s)

        lamb_diag = torch.diag_embed(lamb)

        Weight = torch.matmul(torch.matmul(U, lamb_diag), U.permute(0, 2, 1))

        return Weight

    def weight_wavelet_inverse(self, s, lamb, U):
        s = s
        for i in range(len(lamb)):
            for j in range(len(lamb[0])):
                lamb[i][j] = math.exp(-lamb[i][j] * s)

        lamb = torch.exp(lamb * s)

        lamb_diag = torch.diag_embed(lamb)

        Weight = torch.matmul(torch.matmul(U, lamb_diag), U.permute(0, 2, 1))

        return Weight

    def wavelet_basis(self, s, lamb, U, threshold):

        lamb, U = (torch.from_numpy(lamb)).float().to(self.device), (
            torch.from_numpy(U)).float().to(self.device)

        Weight = self.weight_wavelet(s, lamb, U)
        inverse_Weight = self.weight_wavelet_inverse(s, lamb, U)

        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

        Weight = F.normalize(Weight, p=1, dim=2)
        inverse_Weight = F.normalize(inverse_Weight, p=1, dim=2)

        return Weight, inverse_Weight

    def fast_wavelet_basis(self, adj, s, threshold, m):

        L = self.laplacian_multi(adj)

        lamb = self.largest_lamb(L)

        a = lamb / 2
        c = []
        inverse_c = []
        for j in range(len(a)):
            Temp_c = []
            Temp_inverse_c = []
            for i in range(m + 1):  #
                f_res = 2 * np.exp(s * a[j]) * iv(i, s * a[j])
                inverse_f_res = 2 * np.exp(-s * a[j]) * iv(i, -s * a[j])
                Temp_c.append(f_res)
                Temp_inverse_c.append(inverse_f_res)
            c.append(Temp_c)
            inverse_c.append(Temp_inverse_c)

        c = torch.tensor(c).unsqueeze(-1).unsqueeze(-1).to(self.device)
        inverse_c = torch.tensor(inverse_c).unsqueeze(-1).unsqueeze(-1).to(self.device)

        L = L.to(self.device)

        L_cheb = self.cheb_polynomial_multi(L).to(self.device)  # 切比雪夫四阶近似 (706，4,41,41)

        # print(f"Shape of c: {c.shape}")
        # print(f"c: {c}")
        # print(f"Shape of L_cheb: {L_cheb.shape}")

        Weight = torch.sum(c * L_cheb, dim=1)

        # 计算 inverse_Weight
        inverse_Weight = torch.sum(inverse_c * L_cheb, dim=1)

        # Weight[Weight < threshold] = 0.0
        # inverse_Weight[inverse_Weight < threshold] = 0.0

        Weight = F.normalize(Weight, p=1, dim=2)
        inverse_Weight = F.normalize(inverse_Weight, p=1, dim=2)

        return Weight.float(), inverse_Weight.float()

    def cheb_polynomial_multi(self, laplacian):  # 返回多阶拉普拉斯矩阵,这里使用的切比雪夫不等式的四阶式子
        # print('laplacian.shape '+str(laplacian.shape)) #torch.Size([145, 41, 41])
        bat, N, N = laplacian.size()  # [N, N] 512
        laplacian = laplacian.unsqueeze(1)
        first_laplacian = torch.zeros([bat, 1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=1)

        return multi_order_laplacian  # bx4x41x41

    def reconstruction_loss(self, rev_input, input):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(rev_input, input)
        return loss.to(self.device)

    def forward(self, G, Loop_G, RNA_OP, RNA_ELMo, RegionIndex): #RegionIndex是对应的RNA样本上筛选出的motif的起始位置和结束位置。

        # print(G.shape) # torch.Size([706, 41, 41])
        # print(Loop_G.shape) # torch.Size([706, 41, 41])
        # print(RNA_OP.shape) # torch.Size([706, 41, 15]) # 理化性质
        # print(RNA_ELMo.shape) # torch.Size([706, 41, 256]) # 语义信息

        final_feature, Constructive_loss, motifGraph = self.RNASequenceFilter(RNA_OP, RegionIndex, RNA_ELMo)

        # RNA_ELMo = self.fc2(self.activation(self.fc1(RNA_ELMo)))
        # RNA_ELMo = RNA_ELMo.mean(dim=1)

        consensus_vectors, distance_sums, consensus_graph = self.NodeConsensusVector(motifGraph)  # consensus_graph为 nxn 的二维张量

        # consensus_motif_graph = consensus_graph + motifGraph  # 引入共识知识
        # consensus_motif_graph = motifGraph  # 引入共识知识
        consensus_motif_graph = consensus_graph # 引入共识知识

        consensus_motif_graph = consensus_motif_graph.unsqueeze(0)

        wavelets, wavelet_inv = self.fast_wavelet_basis(consensus_motif_graph, 0.3, 1e-4, 3)  # 小波卷积
        # print(wavelets)

        # mul_L = self.laplacian_multi(consensus_motif_graph)

        final_representation = self.graph_wavelet_layer(final_feature, wavelets, wavelet_inv)

        # final_representation =torch.matmul(mul_L,final_feature)
        # final_representation = final_feature

        # 识别过程（标准化过个全连接层，再接个分类器）
        final_representation = self.bn1(final_representation.squeeze())
        # print(final_representation.shape)

        scores = self.prob(final_representation)
        # print(scores.shape) # torch.Size([706, 1])
        # print(distance_sums)
        # print(Constructive_loss)
        print(final_representation.shape)

        return final_representation, scores, Constructive_loss, distance_sums
