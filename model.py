import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalGAT(nn.Module):
    def __init__(self, dataset, type_num, length=32, output=8, aggregate_method="max"):
        """_summary_

        Args:
            dataset (Dataset): the dataset you will train.
            type_num (int, optional): Num of types if in embedding mode.
            length (int): num of length. Defaults to 32.
            output (int): output dim. Defaults to 8.
            aggregate_method (str, optional): The method how the causal matrix aggeragate. Defaults to 'max'.
        """
        super(CausalGAT, self).__init__()

        self.output = output
        self.embedding = nn.Embedding(type_num, 6 * length)
        nn.init.xavier_uniform_(self.embedding.weight, gain=1.44)

        self.GAT = GraphAttentionLayer(
            dataset.attributor_num,
            length,
            dataset.initial_weight.to(DEVICE),
            aggregate_method,
        )
        self.project_1 = nn.Linear(2 * length, 4 * output)
        nn.init.xavier_uniform_(
            self.project_1.weight, gain=nn.init.calculate_gain("relu")
        )
        self.batchnorm = nn.BatchNorm1d(4 * output)
        self.relu = nn.ReLU()
        self.project_2 = nn.Linear(4 * output, 2 * output)
        nn.init.xavier_uniform_(self.project_2.weight, gain=1.44)

        self.adj_tensor = dataset.adj_tensor
        self.ts_count = dataset.ts_count

    def __forward(self, x, adj):
        main = x[:, [0, -1]].to(torch.long)  # shape (batch_size, 2)
        attributor = x[:, 1:-1].unsqueeze(1)  # shape (batch_size, 1, attributor_num)
        user, item = torch.chunk(
            self.embedding(main).mT, 2, 2
        )  # shape (batch_size, 6*length, 1)
        effect_0 = self.GAT(
            user, item, attributor, adj
        )  # shape (batch_size, length, 2)

        effect = self.relu(
            self.batchnorm(self.project_1(effect_0.flatten(1)))
        )  # shape (batch_size, 4*output)

        user_effect, item_effect = torch.chunk(
            self.project_2(effect).reshape(-1, self.output, 2), 2, 2
        )  # shape (batch_size, output, 1)
        return user_effect, item_effect

    def forward(self, x, x_disturbed, adj):
        user_effect, item_effect = self.__forward(x, adj)
        user_disturbed_effect, item_disturbed_effect = self.__forward(x_disturbed, adj)
        user_similarity = (
            user_effect.mT
            @ user_disturbed_effect
            / torch.norm(user_effect, dim=1, keepdim=True)
            / torch.norm(user_disturbed_effect, dim=1, keepdim=True)
        )
        item_similarity = (
            item_effect.mT
            @ item_disturbed_effect
            / torch.norm(item_effect, dim=1, keepdim=True)
            / torch.norm(item_disturbed_effect, dim=1, keepdim=True)
        )

        return user_similarity + item_similarity

    def map(self, original_feature, adj):
        # shape (N, user + original_feature_num + item)
        with torch.no_grad():
            main = original_feature[:, [0, -1]].to(torch.long)
            attributor = original_feature[:, 1:-1].unsqueeze(
                1
            )  # shape (N, 1, original_feature_num)
            user, item = torch.chunk(
                self.embedding(main).mT, 2, 2
            )  # shape (N, 6*length, 1)
            return self.GAT(user, item, attributor, adj)  # shape (N, length, 2)


class GraphAttentionLayer(nn.Module):
    def __init__(
        self, attributor_num, length=32, initial_weight=None, aggregate_method="max"
    ):
        """Graph Attention Layer

        Args:
            attributor_num (int): Num of attributors.
            length (int, optional): QKV length. Defaults to 32.
            initial_weight (int, optional): Initial_weight of interation. Defaults to None.
        """
        super(GraphAttentionLayer, self).__init__()
        self.length = length
        self.initial_weight = initial_weight  # shape (N, N)
        self.causal_matrix = torch.where(
            initial_weight > 0, torch.ones_like(initial_weight) / 2, 0
        )
        self.aggregate_method = aggregate_method

        self.W_1 = nn.Parameter(torch.zeros(size=(6 * length, attributor_num)))
        nn.init.xavier_uniform_(self.W_1.data, gain=nn.init.calculate_gain("relu"))

        self.relu = nn.ReLU()
        self.W_2 = nn.Parameter(torch.zeros(size=(6 * length, length)))
        nn.init.xavier_uniform_(self.W_2.data, gain=1.44)

    def forward(self, user, item, attributor, adj):

        attributor = (
            attributor * self.W_1
        )  # shape (batch_size, 6*length, attributor_num)

        raw_attention = torch.cat(
            [user, attributor, item], dim=2
        )  # shape (batch_size, 6*length, N)
        Query, Key, Value = torch.chunk(
            raw_attention, 3, 1
        )  # shape (batch_size, 2*length, N)

        N = self.length
        Query = (self.relu(Query).transpose(1, 2) @ self.W_2[: 2 * N]).transpose(
            1, 2
        )  # shape (batch_size, length, N)
        Key = (self.relu(Key).transpose(1, 2) @ self.W_2[2 * N: 4 * N]).transpose(
            1, 2
        )  # shape (batch_size, length, N)
        Value = (self.relu(Value).transpose(1, 2) @ self.W_2[4 * N:]).transpose(
            1, 2
        )  # shape (batch_size, length, N)

        h = (Query.mT @ Key) * adj * self.initial_weight   # shape (batch_size, N, N)
        exp_h = torch.exp(torch.where(h - h.mT > 0, 0, h - h.mT))  # shape (batch_size, N, N)
        causal_matrix = exp_h / (exp_h + exp_h.mT) * adj  # shape (batch_size, N, N)

        if self.aggregate_method == "max":
            self.causal_matrix = causal_matrix.max(dim=0)[0]
        elif self.aggregate_method == "mean":
            self.causal_matrix = causal_matrix.mean(dim=0)

        Ans = Value @ causal_matrix  # shape (batch_size, length, N)

        return Ans[:, :, [0, -1]]  # shape (batch_size, length, 2)
