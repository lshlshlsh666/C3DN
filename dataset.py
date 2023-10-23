import torch
from torch.utils.data import Dataset, DataLoader
from utils import Constrained


class dataset(Dataset):
    def __init__(self, X, mode="normal"):
        """The dataset used in the CGFTP model.
        Args:
            X (np.array/torch.Tensor): The data matrix. The shape is (N, M), where N is the number of samples,
                            and M is the number of features. M = user_id + attributor_num + item_id + timestamp.
        """

        def list2tensor(X_list, max_length):
            X_tensor = torch.zeros(len(X_list), max_length, X_list[0].shape[1] - 1)
            for k, X in enumerate(X_list):
                X_tensor[k] = torch.nn.functional.pad(
                    X[:, :-1], (0, 0, 0, max_length - X.shape[0])
                )
            return X_tensor

        if type(X) != torch.Tensor:
            X = torch.as_tensor(X)
        self.X = X.to(torch.float32)
        self.attributor_num = X.shape[-1] - 3

        self.ts_list, self.ts_count = self.X[:, -1].unique(return_counts=True)
        self.X_list = [X[X[:, -1] == ts] for ts in self.ts_list]
        self.X_tensor = list2tensor(self.X_list, self.ts_count.max()).to(torch.float32)
        self.ts_num = len(self.ts_list)
        self.uer_item_type_num = 0

        adj_list = []
        print(
            "Start use UG to calculate the pre_causality adjoint matrix for each timestep:"
        )
        for k in range(self.ts_num):

            print(f"Now {k}".rjust(20) + f"/{self.ts_num}".rjust(5), end="\r")

            # attributer_adj = torch.as_tensor(
            #     Constrained.UG(self.X_list[k][:, 1:-2], mode=mode)
            # )
            # attributer_sort = (attributer_adj.sum(0).argsort(descending=True))[
            #     : self.attributor_num // 5
            # ]
            # user_item_adj = torch.zeros(self.attributor_num)
            # user_item_adj[attributer_sort] = 1
            # adj = Constrained.argument(attributer_adj, user_item_adj)
            
            adj = Constrained.UG(self.X_list[k][:, :-1], mode=mode)

            assert adj.shape == (self.attributor_num + 2, self.attributor_num + 2)

            adj_list.append(torch.as_tensor(adj).unsqueeze(0))

        self.adj_tensor = torch.cat(adj_list, 0).to(torch.float32)

        self.initial_weight = self.adj_tensor.sum(0) / self.ts_num

        self.in_group_index = torch.cat(
            [torch.arange(ts_num) for ts_num in self.ts_count]
        )
        self.ts_index = torch.repeat_interleave(
            torch.arange(self.ts_num), self.ts_count
        )
        disturb_index = torch.randint(0, 2, [self.X.shape[0]]) * 2 - 1
        disturb_index[: self.ts_count[0]] = 1
        disturb_index[self.ts_count[-1]:] = -1
        self.disturb_ts_index = self.ts_index + disturb_index

    def __getitem__(self, index):
        return (
            self.in_group_index[index],
            self.ts_index[index],
            self.disturb_ts_index[index],
        )

    def __len__(self):
        return len(self.X)

    def loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)
