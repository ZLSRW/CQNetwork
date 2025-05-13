
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # 确保导入了 numpy
import re
import random

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class SequenceSplitter(nn.Module):
    def __init__(self, seq_length=41, target_length=8, embedding_dim=256, reduced_dim=64, device=None):
        super(SequenceSplitter, self).__init__()
        self.seq_length = seq_length
        self.target_length = target_length
        self.embedding_dim = embedding_dim
        self.reduced_dim = reduced_dim
        self.device = device

        # 定义两个全连接层
        self.fc_pos = nn.Sequential(
            nn.Linear(embedding_dim, reduced_dim),
            nn.Tanh(),
            nn.Linear(reduced_dim, reduced_dim),
            nn.Tanh(),
        )
        self.fc_neg = nn.Sequential(
            nn.Linear(embedding_dim, reduced_dim),
            nn.Tanh(),
            nn.Linear(reduced_dim, reduced_dim),
            nn.Tanh(),
        )

        self.fc_all = nn.Sequential(
            nn.Linear(embedding_dim, reduced_dim),
            nn.Tanh(),
        )

        self.Pos_weights = None
        self.Neg_weights = None

    def _initial_Pos_weights(self, n, seq_length):
        self.Pos_weights = nn.Parameter(torch.ones((n, seq_length), requires_grad=True)).to(self.device)
        return

    def _initial_Neg_weights(self, n, seq_length):
        self.Neg_weights = nn.Parameter(torch.ones((n, seq_length), requires_grad=True)).to(self.device)
        return

    def compute_similarity(self, pos_tensor, neg_tensor):
        # 计算正张量之间的欧几里得距离
        pos_dist = torch.cdist(pos_tensor, pos_tensor, p=2)
        pos_sim_value = torch.exp(-pos_dist).sum()

        # 计算正张量和负张量之间的欧几里得距离
        neg_dist = torch.cdist(pos_tensor, neg_tensor, p=2)
        neg_sim_value = torch.exp(-neg_dist).sum()

        return pos_sim_value, neg_sim_value

    def forward(self, C, L):

        n, _, embedding_dim = L.shape

        # L的部分需要融入第一部分中二级结构和一级结构的权重，以引入一级结构和二级结构见解。
        if self.Pos_weights is None:
            self._initial_Pos_weights(L.size(0), L.size(1))
        if self.Neg_weights is None:
            self._initial_Neg_weights(L.size(0), L.size(1))

        # 初始化正张量和负张量
        positive_tensor = torch.zeros((n, self.target_length, embedding_dim), device=L.device)
        negative_tensor = torch.zeros((n, self.seq_length - self.target_length, embedding_dim), device=L.device)
        pos_weights_mask = torch.zeros_like(self.Pos_weights)
        neg_weights_mask = torch.zeros_like(self.Neg_weights)

        for i in range(n):
            start, end = C[i, 0].item(), C[i, 1].item()
            pos_region = L[i, start:end + 1, :]

            # 对self.Pos_weights进行mask
            pos_weights_mask[i, start:end + 1] = self.Pos_weights[i, start:end + 1]

            # 确保正张量的长度为target_length
            if pos_region.shape[0] == 0:
                pos_region = torch.zeros((self.target_length, embedding_dim), device=L.device)
            elif pos_region.shape[0] != self.target_length:
                pos_region = F.interpolate(pos_region.unsqueeze(0).transpose(1, 2), size=(self.target_length),
                                           mode='linear', align_corners=False).squeeze(0).transpose(0, 1)

            positive_tensor[i] = pos_region

            # 获取负张量的部分
            neg_indices = list(range(0, start)) + list(range(end + 1, self.seq_length))
            neg_region = L[i, neg_indices, :]

            # 对self.Neg_weights进行mask
            neg_weights_mask[i, neg_indices] = self.Neg_weights[i, neg_indices]

            # 确保负张量的长度为seq_length - target_length
            if len(neg_indices) == 0:
                neg_region = torch.zeros((self.seq_length - self.target_length, embedding_dim), device=L.device)
            elif len(neg_indices) != (self.seq_length - self.target_length):
                neg_region = F.interpolate(neg_region.unsqueeze(0).transpose(1, 2),
                                           size=(self.seq_length - self.target_length), mode='linear',
                                           align_corners=False).squeeze(0).transpose(0, 1)

            negative_tensor[i] = neg_region

        # 对正张量和负张量进行形状变化和全连接层处理
        Pos_representation = pos_weights_mask[:L.size(0), :].unsqueeze(-1) * L
        # print(Pos_representation.shape)
        Neg_representation = neg_weights_mask[:L.size(0), :].unsqueeze(-1) * L

        Key_tensor = self.fc_pos(Pos_representation)
        pos_tensor = Key_tensor.mean(dim=1)

        Non_Key_tensor = self.fc_neg(Neg_representation)
        neg_tensor = Non_Key_tensor.mean(dim=1)

        pos_sim_value, neg_sim_value = self.compute_similarity(pos_tensor, neg_tensor)

        # 计算对比损失
        Constructive_loss = -torch.log(pos_sim_value / neg_sim_value)

        # 对正样本和负样本的特征进行融合，作为节点的特征。
        # final_feature = self.fc_all(Key_tensor+Non_Key_tensor).mean(dim=1)
        final_feature = self.fc_all(Pos_representation + Neg_representation).mean(dim=1)

        return final_feature, Constructive_loss


# # 示例代码
# L = torch.randn((10, 41, 256))  # 假设L是一个形如(n, 41, 256)的张量
# C = torch.randint(0, 34, (10, 2))  # 假设C是一个形如(n, 2)的张量，其中每个元素为(起始下标, 结束下标)
#
# splitter = SequenceSplitter()
# positive_tensor, negative_tensor, loss = splitter(L, C)
#
# print("正张量形状：", positive_tensor.shape)  # 预期输出 (n, 8, 256)
# print("负张量形状：", negative_tensor.shape)  # 预期输出 (n, 33, 256)
# print("对比损失：", loss.item())

class RNASequenceFilter(nn.Module):  # 发现关键子序列并构建motif图
    def __init__(self, vector_length=15, embedding_dim=32, target_length=8, device=None):
        super(RNASequenceFilter, self).__init__()
        self.vector_length = vector_length
        self.embedding_dim = embedding_dim
        self.target_length = target_length
        self.device = device

        self.sigma = nn.Parameter(torch.tensor(1.0))  # 学习的尺度参数

        # 全连接层，将关键子区域的向量矩阵转换为长度为32的一维向量
        self.fc = nn.Sequential(
            nn.Linear(vector_length * target_length, embedding_dim),
            nn.Tanh(),
            nn.BatchNorm1d(embedding_dim)
        )

        self.NPfeature = SequenceSplitter(device=self.device)

        self.to('cuda:0')

    def build_graph(self, A, C):
        n = A.shape[0]
        start_positions = C[:, 0]
        end_positions = C[:, 1]
        key_regions = torch.zeros((n, self.target_length, self.vector_length))

        for i in range(n):
            start, end = start_positions[i].item(), end_positions[i].item()
            vec = A[i, start:end + 1, :]
            if vec.shape[0] != self.target_length:
                vec = vec.permute(1, 0).unsqueeze(0)
                vec = F.interpolate(vec, size=self.target_length, mode='linear', align_corners=False)
                vec = vec.squeeze(0).permute(1, 0)
            key_regions[i] = vec

        key_regions_flat = key_regions.view(n, self.target_length * self.vector_length)
        key_regions_flat = key_regions_flat.to(self.device)
        embeddings = self.fc(key_regions_flat)
        D = torch.exp(-torch.cdist(embeddings, embeddings, p=2) / (2 * self.sigma ** 2)).clone()
        # D = torch.exp(-torch.cdist(embeddings, embeddings, p=2) / 2).clone()
        mask = torch.ones_like(D)
        mask.fill_diagonal_(0)
        non_diag_elements = D[mask.bool()]
        avg_val = non_diag_elements.mean()
        D[D < avg_val] = 0.0
        non_diag_elements = D[mask.bool()]
        min_val = non_diag_elements.min()
        max_val = non_diag_elements.max()

        if max_val == min_val:
            max_val = min_val + 1

        D = (D - min_val) / (max_val - min_val)
        D.fill_diagonal_(1)

        # print(D)

        return D

    def identify_key_regions(self, A, B, motifs):
        n, _, _ = A.shape
        C = torch.zeros((n, 2), dtype=torch.int)

        for i in range(n):
            sequence = ''.join(['A' if A[i, j, 0] == 1 else
                                'C' if A[i, j, 1] == 1 else
                                'G' if A[i, j, 2] == 1 else
                                'U' if A[i, j, 3] == 1 else 'N' for j in range(41)])
            structure_matrix = B[i]

            max_score = float('-inf')
            best_start, best_end = 0, 0
            found_match = False

            for start in range(34):
                for end in range(start + 1, min(start + 9, 41)):
                    sub_seq = sequence[start:end]
                    sub_matrix = structure_matrix[start:end, start:end]

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
                best_start = random.randint(0, 33)
                best_end = best_start + 7

            C[i, 0] = best_start
            C[i, 1] = best_end

        return C

    def forward(self, A, C, L):  # onehot理化性质，局部区域的前后坐标，ELMo特征

        n = A.size(0)
        motifGraph = self.build_graph(A, C)
        # L = L * (self.motif_weights[:A.size(0), :] + self.structure_weights[:A.size(0), :]).unsqueeze(-1)

        # print(self.motif_weights)

        final_feature, Constructive_loss = self.NPfeature(C, L)

        motifGraph = motifGraph.to(self.device)

        return final_feature, Constructive_loss, motifGraph


# # 示例优化过程
# # 示例数据
# n = 5
# A = torch.rand((n, 41, 15))  # 随机生成
# B = torch.randint(0, 2, (n, 41, 41))  # 随机生成
# L = torch.randn((n, 41, 256))
#
# # 假设 A 的 one-hot 编码在前4位
# A[:, :, :4] = F.one_hot(torch.randint(0, 4, (n, 41)), num_classes=4).float()
#
# # 定义模型
# model = RNASequenceFilter(n)
#
# # 示例优化过程
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = torch.nn.MSELoss()
#
# # 假设我们有一些目标值
# target = torch.randint(0, 41, (n, 2))
#
# # 将 target 转换为浮点数
# target = target.float()
#
# output, D, x = model(A, B, L)

from scipy.sparse.csgraph import floyd_warshall

torch.autograd.set_detect_anomaly(True)  # 启用异常检测


class NodeConsensusVector(nn.Module):
    def __init__(self, num_anchor_points=100, hidden_size=10, device=None):
        super(NodeConsensusVector, self).__init__()

        # 确定锚点集的大小，默认为motifGraph总节点数的20%
        self.num_anchor_points = num_anchor_points
        self.anchor_points = None

        # 初始化可学习权重，范围在0到1之间
        self.node_weights = None
        self.weight_matrix = None

        # 初始化全连接层
        self.fc = nn.Linear(self.num_anchor_points, hidden_size)
        self.activation = nn.Tanh()
        self.device = device

        self.sigma = nn.Parameter(torch.tensor(1.0))

        self.to(self.device)

    def _initialize_weights(self, num_nodes):
        # 初始化可学习权重
        self.node_weights = nn.Parameter(torch.rand(num_nodes)).to(self.device)

    def _initialize_weights_vectors(self, num_nodes):
        # 初始化可学习权重矩阵
        self.weight_matrix = nn.Parameter(torch.rand(num_nodes, self.num_anchor_points).to(self.device))

    def _adjust_motif_graph(self, motifGraph, n):
        # 使用权重调整 motifGraph 的每一行
        adjusted_motifGraph = motifGraph * self.node_weights[:n].unsqueeze(1)
        return adjusted_motifGraph

    def _select_anchor_points(self, num_nodes, MaxIndex):
        # 使用权重选择锚点集，选择权重最大的前10%的节点
        if self.node_weights is None:
            self._initialize_weights(100)

        weights = self.node_weights.cpu().detach().numpy()
        # num_top_points = max(1, int(num_nodes * 0.1))  # 确保至少选择一个节点
        num_top_points = max(1, 100)  # 确保至少选择一个节点

        # 获取所有节点的权重排序索引
        sorted_indices = np.argsort(weights)[::-1]  # 从大到小排序
        top_indices = sorted_indices[:num_top_points]  # 选择前10%的节点

        # 确保选出的节点索引在 MaxIndex 范围内
        valid_indices = top_indices[top_indices < MaxIndex]

        # 如果有效的索引数量少于 num_top_points，则补充索引直到数量满足 num_top_points
        if len(valid_indices) < num_top_points:
            remaining_indices = sorted_indices[~np.isin(sorted_indices, valid_indices)]
            additional_indices = remaining_indices[remaining_indices < MaxIndex]
            valid_indices = np.concatenate([valid_indices, additional_indices[:num_top_points - len(valid_indices)]])

        self.anchor_points = valid_indices

    def _calculate_distances(self, motifGraph):
        # 转换为 csr 矩阵以便于 scipy 处理
        motifGraph_csr = csr_matrix(motifGraph.cpu().detach().numpy())

        # 使用 Floyd-Warshall 算法计算所有节点之间的最短路径矩阵
        dist_matrix = floyd_warshall(motifGraph_csr, directed=False)

        dist_matrix[np.isinf(dist_matrix)] = 100

        # print(dist_matrix.shape)  # (706, 706)

        return dist_matrix

    # def build_consensus_graph(self, consensus_vectors, sigma=1.0):
    #
    #     # activated_vectors = self.fc(consensus_vectors)
    #     activated_vectors = consensus_vectors
    #     # 计算每对向量之间的欧氏距离
    #     pairwise_distances = torch.cdist(activated_vectors, activated_vectors, p=2)
    #     # print(pairwise_distances)
    #
    #     # 放大差异，通过对距离的平方进行缩放
    #     # 对距离的平方进行缩放可以放大相似性差异
    #     scaled_distances = pairwise_distances ** 4 # 大于1的相似性越大，小于1的相似性越小，可以增大差异
    #     # 使用高斯核函数计算相似性
    #     # exp(-dist^2 / (2 * sigma^2))
    #     similarity_matrix = torch.exp(-scaled_distances / (2 * sigma ** 2))
    #
    #     # 让矩阵对称（虽然高斯核函数本身是对称的，但为了确保对称性）
    #     # similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
    #
    #     return similarity_matrix

    def build_consensus_graph(self, consensus_vectors):
        # 计算每对向量之间的欧氏距离
        pairwise_distances = torch.cdist(consensus_vectors, consensus_vectors, p=2)

        # 放大差异，通过对距离的平方进行缩放
        scaled_distances = pairwise_distances ** 4

        # 使用高斯核函数计算相似性
        similarity_matrix = torch.exp(-scaled_distances / (2 * self.sigma ** 2)).clone()

        return similarity_matrix

    def forward(self, motifGraph):
        num_nodes = motifGraph.shape[0]

        if self.node_weights is None:
            self._initialize_weights(num_nodes)  # 初始化权重
        if self.weight_matrix is None:
            self._initialize_weights_vectors(num_nodes)

        self._select_anchor_points(num_nodes, num_nodes)  # 选择锚点集

        # 调整 motifGraph 的每一行
        adjusted_motifGraph = self._adjust_motif_graph(motifGraph, motifGraph.size(0))

        # 计算所有节点之间的最短路径矩阵
        dist_matrix = self._calculate_distances(adjusted_motifGraph)

        anchor_indices = np.array(self.anchor_points)

        # print(dist_matrix.shape)
        # print(anchor_indices.shape)

        consensus_vectors = torch.tensor(dist_matrix[:, anchor_indices], dtype=torch.float32, device=self.device)

        distance_sums = consensus_vectors.sum(dim=1) - 100
        distance_sums = distance_sums.sum(dim=0) / (num_nodes * self.num_anchor_points)

        print(consensus_vectors.shape)
        print(self.weight_matrix[:motifGraph.size(0), :].shape)

        weighted_consensus_vectors = self.weight_matrix[:motifGraph.size(0), :] * consensus_vectors  # 这个权重矩阵需要参与优化

        # 构建共识图
        consensus_graph = self.build_consensus_graph(weighted_consensus_vectors)

        return consensus_vectors, distance_sums, consensus_graph

# # 创建一个模拟的 41x41 motifGraph
# num_nodes = 41
# motifGraph = torch.rand((num_nodes, num_nodes))
#
# # 初始化 NodeConsensusVector 类
# num_anchor_points = max(1, num_nodes // 10)  # 选择最多 10% 的节点作为锚点
# node_consensus_vector = NodeConsensusVector(num_anchor_points=num_anchor_points)
#
# # 计算共识向量矩阵和距离之和
# consensus_vectors, distance_sums, consensus_graph = node_consensus_vector(motifGraph)
#
# # 打印结果
# print("Motif Graph:")
# print(motifGraph)
# print("\nAdjusted Motif Graph:")
# print(node_consensus_vector._adjust_motif_graph(motifGraph))
# print("\nConsensus Vectors:")
# print(consensus_vectors)
# print("\nDistance Sums:")
# print(distance_sums)
# print("\nNode Weights:")
# print(node_consensus_vector.node_weights)
# print("\nconsensus graph:")
# print(consensus_graph)
