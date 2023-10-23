# The function used in other module.
import numpy as np
import torch
import networkx as nx
from scipy.stats import norm
from itertools import combinations

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Constrained:
    """The class of the constrained graph structure learning algorithm."""

    def __init__(self):
        pass

    def UG(X, alpha=0.005, initial=None, mode="fast"):
        """The algorithm of the UG algorithm.
        Args:
            X (np.array): The sample matrix.
            alpha (float in [0,1]): the p-value of the statistic test.
            initial (np.array, optional): The initial adjoint matrix. Defaults to None.
            mode (str, optional): The mode of the algorithm, normal means traditional PC algorithm, 'fast' means the fast version of the PC algorithm. Defaults to 'normal'.
        """

        def conditional_independet_discovery(i, j, k):
            """Test the conditional independency between the r.v. i and j.
            Args:
                i (int): the r.v. which we need to test the partial correlation.
                j (int): the r.v. which we need to test the partial correlation.
                k (int): the length of the conditional varible set.
            """

            def Fisher_Z(K):
                """Calculate the partial correlation coeffecients and transfrom it into the
                Normal distribution through the Fisher_Z transformation.

                Args:
                    K (set): the list of the conditional varibles.

                """
                K = list(K)
                cov_matrix = C[np.ix_([i, j] + K, [i, j] + K)]
                pinv_cov_matrix = np.linalg.pinv(
                    cov_matrix
                )  # the pseudo inverse of the covariance matrix.
                rho = (
                    -1
                    * pinv_cov_matrix[0, 1]
                    / (
                        np.sqrt(abs(pinv_cov_matrix[0, 0] * pinv_cov_matrix[1, 1]))
                        + 1e-10
                    )
                )
                rho = min(
                    1 - 1e-9, max(rho, -1 + 1e-9)
                )  # The range of the partial correlation coeffecients is [-1,1].
                Z = np.arctanh(rho)  # The Fisher_Z transformation.
                Z_norm_abs = np.sqrt(sample_size - len(K) - 3) * abs(
                    Z
                )  # Since the distrbution is symmetric, we use abs().

                threshod = norm.ppf(1 - alpha / 2)

                return Z_norm_abs <= threshod  # independent

            K_list = combinations(i_adjoint_list, k)
            for corr_set in K_list:
                if Fisher_Z(corr_set):
                    return True
            return False

        n = np.size(X, 1)  # The num of the nodes.
        sample_size = np.size(X, 0)  # the num of samples.
        C = np.cov(X.T)  # the correlation matrix of the sample matrix
        if initial is not None:
            A = initial
        else:
            A = (np.ones((n, n)) - np.eye(n)).astype(int)  # Adjoint matrix.
        D = A.sum(axis=1)  # The degree of each node.
        if mode == "normal":
            num_of_condition_set = -1
            while sample_size - num_of_condition_set - 4 > 0:
                num_of_condition_set += 1
                for i, row in enumerate(A[-1]):
                    for j, A_ij in enumerate(row[:i]):
                        print(f"{i*n+j}".ljust(7)+f"/{n*n}", end="\r")
                        i_adjoint_list = [n for n, x in enumerate(A[i]) if x]
                        if A_ij and (len(i_adjoint_list) >= num_of_condition_set + 1):
                            i_adjoint_list.remove(j)
                            # if conditional_independet_discovery(i, j, num_of_condition_set) and (D[i] != 1) and (D[j] != 1):
                            if conditional_independet_discovery(
                                i, j, num_of_condition_set
                            ):
                                A[i][j] = 0
                                A[j][i] = 0
                                D[i] -= 1
                                D[j] -= 1
                if num_of_condition_set >= np.max(D) - 1:
                    break
        elif mode == "fast":
            for i, row in enumerate(A):
                for j, A_ij in enumerate(row):
                    print(f"{i*n+j}".ljust(7)+f"/{n*n}", end="\r")
                    i_adjoint_list = [n for n, x in enumerate(A[i]) if x]
                    if A_ij and (len(i_adjoint_list) > 1):
                        i_adjoint_list.remove(j)
                        if conditional_independet_discovery(i, j, len(i_adjoint_list)):
                            A[i][j] = 0
                            A[j][i] = 0
                            D[i] -= 1
                            D[j] -= 1
        return A.astype(int)

    def argument(A, a):
        """
        Given an n-2 dim matrix and an n-2 dim vector, return the argument n dim matrix. 
        """
        n = len(a)
        arg_A = torch.cat([a.unsqueeze(0), A, a.unsqueeze(0)], dim=0)
        arg_a = torch.cat([torch.zeros(1), a, torch.zeros(1)])
        return torch.cat([arg_a.unsqueeze(1), arg_A, arg_a.unsqueeze(1)], dim=1)
    
    def drawnet(A, ax=None):
        """Draw the network.
        Args:
            A (np.array): The adjoint matrix of the network.
            ax (axis, optional): The axis of the figure. Defaults to None.
        """
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        pos = nx.random_layout(G)
        nx.draw_networkx(G, pos=pos, ax=ax)

class SVD_GFT:
    """The class of the SVD_GFT algorithm."""

    def __init__(self, dataset, disturb_dim=1, disturbance_max_ratio=0.5):
        """
        Args:
            dataset (torch.dataset): the dataset you are using (train/valid/test).
            disturb_dim (int, optional): : the num of the lower frequency you want to disturb. (Start from lowest frequency) Defaults to 1.
            disturb_strength (float, optional): disturbance max ratio. Defaults to 0.5.
        """
        self.dataset = dataset
        self.disturb_dim = disturb_dim
        self.disturbance_max_ratio = disturbance_max_ratio

    def __Disturbance(self, X1, X2, latest_causal_matrix):
        """Disturbance the lower frequency of the signal using the adjacent timestamp.

        Args:
            X1 (torch.Tensor): the signal of the first timestamp.
            X2 (torch.Tensor): the signal of the second timestamp.
            latest_causal_matrix (torch.Tensor): the latest causality matrix.
        """

        def SVD(A):
            K = A.sum(1).unsqueeze(1) - A
            u, _, v = torch.svd(K)
            return u, v  # shape (len(A), len(A))

        def DiGFT(X, u, v):
            return torch.cat(
                [X @ (u + v) / 2, X @ (u - v) / 2], 1
            )  # shape (batch_size, 2*len(A))

        def DiIGFT(Z, u, v):
            Z1, Z2 = Z[:, :Z.shape[1] // 2], Z[:, Z.shape[1] // 2:]
            return ((Z1 + Z2) @ u.T + (Z1 - Z2) @ v.T) / 2  # shape (batch_size, len(A))

        u, v = SVD(latest_causal_matrix)
        X1_spectrum = DiGFT(X1, u, v)
        X2_spectrum = DiGFT(X2, u, v)  # shape (batch_size, 2*len(A))
        m = latest_causal_matrix.shape[0]

        positive_index = torch.cat(
            [
                torch.arange(m - self.disturb_dim, m),
                torch.arange(2 * m - self.disturb_dim, 2 * m),
            ]
        )
        negative_index = torch.cat(
            [
                torch.arange(self.disturb_dim),
                torch.arange(m, m + self.disturb_dim),
            ]
        )
        alpha_max = self.disturbance_max_ratio
        alpha = (
            torch.rand(X1.shape[0], 1, device=DEVICE) * alpha_max
        )  # shape (batch_size, 1)
        X1_spectrum_copy = X1_spectrum.clone()
        X1_spectrum[:, positive_index] = (1 - alpha) * X1_spectrum[
            :, positive_index
        ] + alpha * X2_spectrum[:, positive_index]
        X1_spectrum_copy[:, negative_index] = (1 - alpha) * X1_spectrum_copy[
            :, negative_index
        ] + alpha * X2_spectrum[:, negative_index]
        
        return DiIGFT(X1_spectrum, u, v),  DiIGFT(X1_spectrum_copy, u, v)  # shape (batch_size, len(A))

    def disturbance(
        self, ingroup_index, ts_index, adjacent_ts_index, latest_causal_matrix
    ):
        index = ingroup_index.repeat(latest_causal_matrix.size()[1], 1).T.unsqueeze(
            1
        )  # shape (batch_size, 1, len(A))
        X1 = (
            self.dataset.X_tensor[ts_index].gather(dim=1, index=index).squeeze(1)
        )  # (batch_size, group_length, N) -> shape (batch_size, len(A))
        X2_index = (
            (
                torch.rand(X1.shape[0], device=DEVICE)
                * self.dataset.ts_count[adjacent_ts_index]
            )
            .floor()
            .long()
            .repeat(latest_causal_matrix.size()[1], 1)
            .T.unsqueeze(1)
        )
        X2 = (
            self.dataset.X_tensor[adjacent_ts_index]
            .gather(dim=1, index=X2_index)
            .squeeze(1)
        )  # shape (batch_size, len(A))
        X1_disturbed_positive, X1_disturbed_negative = self.__Disturbance(
            X1, X2, latest_causal_matrix
        )  # shape (batch_size, len(A))
        X1_disturbed_positive[:, [0, -1]] = X1[:, [0, -1]]
        X1_disturbed_negative[:, [0, -1]] = X1[:, [0, -1]]
        return X1, X1_disturbed_positive, X1_disturbed_negative
