
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

            pos_weights_mask[i, start:end + 1] = self.Pos_weights[i, start:end + 1]

            if pos_region.shape[0] == 0:
                pos_region = torch.zeros((self.target_length, embedding_dim), device=L.device)
            elif pos_region.shape[0] != self.target_length:
                pos_region = F.interpolate(pos_region.unsqueeze(0).transpose(1, 2), size=(self.target_length),
                                           mode='linear', align_corners=False).squeeze(0).transpose(0, 1)

            positive_tensor[i] = pos_region

            neg_indices = list(range(0, start)) + list(range(end + 1, self.seq_length))
            neg_region = L[i, neg_indices, :]


            neg_weights_mask[i, neg_indices] = self.Neg_weights[i, neg_indices]

            if len(neg_indices) == 0:
                neg_region = torch.zeros((self.seq_length - self.target_length, embedding_dim), device=L.device)
            elif len(neg_indices) != (self.seq_length - self.target_length):
                neg_region = F.interpolate(neg_region.unsqueeze(0).transpose(1, 2),
                                           size=(self.seq_length - self.target_length), mode='linear',
                                           align_corners=False).squeeze(0).transpose(0, 1)

            negative_tensor[i] = neg_region

        Pos_representation = pos_weights_mask[:L.size(0), :].unsqueeze(-1) * L
        # print(Pos_representation.shape)
        Neg_representation = neg_weights_mask[:L.size(0), :].unsqueeze(-1) * L

        Key_tensor = self.fc_pos(Pos_representation)
        pos_tensor = Key_tensor.mean(dim=1)

        Non_Key_tensor = self.fc_neg(Neg_representation)
        neg_tensor = Non_Key_tensor.mean(dim=1)

        pos_sim_value, neg_sim_value = self.compute_similarity(pos_tensor, neg_tensor)


        Constructive_loss = -torch.log(pos_sim_value / neg_sim_value)


        # final_feature = self.fc_all(Key_tensor+Non_Key_tensor).mean(dim=1)
        final_feature = self.fc_all(Pos_representation + Neg_representation).mean(dim=1)

        return final_feature, Constructive_loss



class RNASequenceFilter(nn.Module):
    def __init__(self, vector_length=15, embedding_dim=32, target_length=8, device=None):
        super(RNASequenceFilter, self).__init__()
        self.vector_length = vector_length
        self.embedding_dim = embedding_dim
        self.target_length = target_length
        self.device = device

        self.sigma = nn.Parameter(torch.tensor(1.0))


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

    def forward(self, A, C, L):

        n = A.size(0)
        motifGraph = self.build_graph(A, C)
        # L = L * (self.motif_weights[:A.size(0), :] + self.structure_weights[:A.size(0), :]).unsqueeze(-1)

        # print(self.motif_weights)

        final_feature, Constructive_loss = self.NPfeature(C, L)

        motifGraph = motifGraph.to(self.device)

        return final_feature, Constructive_loss, motifGraph

from scipy.sparse.csgraph import floyd_warshall

torch.autograd.set_detect_anomaly(True)

class NodeConsensusVector(nn.Module):
    def __init__(self, num_anchor_points=100, hidden_size=10, device=None):
        super(NodeConsensusVector, self).__init__()

        self.num_anchor_points = num_anchor_points
        self.anchor_points = None

        self.node_weights = None
        self.weight_matrix = None

        self.fc = nn.Linear(self.num_anchor_points, hidden_size)
        self.activation = nn.Tanh()
        self.device = device

        self.sigma = nn.Parameter(torch.tensor(1.0))

        self.to(self.device)

    def _initialize_weights(self, num_nodes):

        self.node_weights = nn.Parameter(torch.rand(num_nodes)).to(self.device)

    def _initialize_weights_vectors(self, num_nodes):

        self.weight_matrix = nn.Parameter(torch.rand(num_nodes, self.num_anchor_points).to(self.device))

    def _adjust_motif_graph(self, motifGraph, n):

        adjusted_motifGraph = motifGraph * self.node_weights[:n].unsqueeze(1)
        return adjusted_motifGraph

    def _select_anchor_points(self, num_nodes, MaxIndex):

        if self.node_weights is None:
            self._initialize_weights(100)

        weights = self.node_weights.cpu().detach().numpy()
        num_top_points = max(1, 100)

        sorted_indices = np.argsort(weights)[::-1]
        top_indices = sorted_indices[:num_top_points]

        valid_indices = top_indices[top_indices < MaxIndex]

        if len(valid_indices) < num_top_points:
            remaining_indices = sorted_indices[~np.isin(sorted_indices, valid_indices)]
            additional_indices = remaining_indices[remaining_indices < MaxIndex]
            valid_indices = np.concatenate([valid_indices, additional_indices[:num_top_points - len(valid_indices)]])

        self.anchor_points = valid_indices

    def _calculate_distances(self, motifGraph):
        motifGraph_csr = csr_matrix(motifGraph.cpu().detach().numpy())

        dist_matrix = floyd_warshall(motifGraph_csr, directed=False)

        dist_matrix[np.isinf(dist_matrix)] = 100


        return dist_matrix


    def build_consensus_graph(self, consensus_vectors):
        pairwise_distances = torch.cdist(consensus_vectors, consensus_vectors, p=2)

        scaled_distances = pairwise_distances ** 4

        similarity_matrix = torch.exp(-scaled_distances / (2 * self.sigma ** 2)).clone()

        return similarity_matrix

    def forward(self, motifGraph):
        num_nodes = motifGraph.shape[0]

        if self.node_weights is None:
            self._initialize_weights(num_nodes)
        if self.weight_matrix is None:
            self._initialize_weights_vectors(num_nodes)

        self._select_anchor_points(num_nodes, num_nodes)


        adjusted_motifGraph = self._adjust_motif_graph(motifGraph, motifGraph.size(0))


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

        consensus_graph = self.build_consensus_graph(weighted_consensus_vectors)

        return consensus_vectors, distance_sums, consensus_graph

