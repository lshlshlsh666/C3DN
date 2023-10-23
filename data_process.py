"""
Make sure the X filled in the following function has the form:
shape: (N, M), where N is the number of samples, and M is the number of features. M = user_id + attributor_num + item_id + timestamp.

Noted that the attubutor contains the edge_feature and the two node_features.
"""

import argparse
from numpy import load, percentile
import torch
import os
from dataset import dataset
os.chdir(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser("The data process")
parser.add_argument(
    "--data", type=str, help="Dataset name (eg. reddit or wikipedia)", default="wikipedia"
)
parser.add_argument(
     "--UG_mode", type=str, help="The mode for UG", default="fast"
 )
parser.add_argument(
    "--valid_size", type=float, help="The ratio of validation set.", default=0.15
)
parser.add_argument(
    "--test_size", type=float, help="The ratio of test set.", default=0.15
)

args = parser.parse_args()

VS = args.valid_size
TS = args.test_size

if __name__ == "__main__":
    X = load(f"Data/{args.data}/pre_processed_{args.data}.npy")
    uer_item_type_num = torch.tensor(X[:, [0, -2]].flatten()).unique().shape[0]
    train_valid_timestamp = percentile(X[:, -1], 100 * (1 - VS - TS))
    valid_test_timestamp = percentile(X[:, -1], 100 * (1 - TS))
    X_train, X_valid, X_test = (
        X[X[:, -1] <= train_valid_timestamp],
        X[(X[:, -1] > train_valid_timestamp) & (X[:, -1] <= valid_test_timestamp)],
        X[X[:, -1] > valid_test_timestamp],
    )
    mode_name = ["train", "valid", "test"]
    for k, data in enumerate([X_train, X_valid, X_test]):
        mode = mode_name[k]
        path = f"Data/{args.data}/{args.data}_dataset_{mode}.pt"
        print(f"Start to save the {mode} dataset.")
        split_dataset = dataset(data, mode=args.UG_mode)
        split_dataset.uer_item_type_num = uer_item_type_num
        torch.save(split_dataset, path)
        print(f"Save the {mode} dataset successfully.")

########################################################################################

# parser = argparse.ArgumentParser("The data process of CGFTP model.")
# parser.add_argument(
#     "--data", type=str, help="Dataset name (eg. DGraphFin)", default="DGraphFin"
# )
# parser.add_argument(
#     "--UG_mode", type=str, help="The mode for UG", default="normal"
# )
# parser.add_argument(
#     "--test_size", type=float, help="The ratio of test set.", default=0.2
# )

# args = parser.parse_args()

# TS = args.test_size

# if __name__ == "__main__":
#     X = load(f"Data/{args.data}/pre_processed_{args.data}.npy", allow_pickle=True)
#     uer_item_type_num = torch.tensor(X[:, [0, -2]].flatten()).unique().shape[0]
#     train_test_timestamp = percentile(X[:, -1], 100 * (1 - TS))
#     X_train, X_test = (
#         X[X[:, -1] < train_test_timestamp],
#         X[(X[:, -1] >= train_test_timestamp)]
#     )
#     mode_name = ["train", "test"]
#     for k, data in enumerate([X_train, X_test]):
#         mode = mode_name[k]
#         path = f"Data/{args.data}/{args.data}_dataset_{mode}.pt"
#         print(f"Start to save the {mode} dataset.")
#         split_dataset = dataset(data, mode=args.UG_mode)
#         split_dataset.uer_item_type_num = uer_item_type_num
#         torch.save(split_dataset, path)
#         print(f"Save the {mode} dataset successfully.")
