import torch
import argparse
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator

# from dataset import dataset
from model import CausalGAT
from utils import SVD_GFT
import time
import os

TIME = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
os.chdir(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser("Train or test the CGFTP model.")
# basic setting
parser.add_argument(
    "--data", type=str, help="Dataset name (eg. DGraphFin)", default="DGraphFin"
)
parser.add_argument("--trail_num", type=int, help="The number of trails.", default=1)
parser.add_argument(
    "--batch_size", type=int, help="The batch size of training.", default=512
)
parser.add_argument("--epoch", type=int, help="The number of epochs.", default=2)

# Disturbance setting
parser.add_argument(
    "--disturbance_dim",
    type=int,
    help="The disturbance dim (Start from the lowest frequency).",
    default=3,
)
parser.add_argument(
    "--disturbance_max_ratio",
    type=float,
    help="The max disturbance ratio.",
    default=0.3,
)
parser.add_argument(
    "--disturbance_step",
    type=float,
    help="The step of updating the cum causal matrix.",
    default=0.05,
)

# CausalGAT setting
parser.add_argument(
    "--causal_gat_length",
    type=int,
    help="the dimension of the Q,K,V in CausalGAT.",
    default=32,
)
parser.add_argument(
    "--causal_gat_output",
    type=int,
    help="the dimention of the output in CausalGAT.",
    default=4,
)
parser.add_argument(
    "--aggeragate_causal_matrix",
    type=str,
    help="the method how the causal matrix aggeragate. (max, mean)",
    default="max",
)

# traing setting
parser.add_argument(
    "--verbose_frequency", type=int, help="The verbose frequency.", default=100
)
parser.add_argument("--lr", type=float, help="The learning rate.", default=5e-3)
parser.add_argument("--optimizer", type=str, help="The optimizer.", default="Adam")
parser.add_argument("--patience", type=int, help="The patience.", default=3)

# Downstream model
parser.add_argument(
    "--trained_model", type=str, help="where is your trained model", default=False
)
parser.add_argument(
    "--downstream_model", type=str, help="The downstream model.", default="GTN"
)

# loading the argparams
try:
    args = parser.parse_args()
except Exception:
    parser.print_help()
    sys.exit(0)

DATA = args.data
TRAIL_NUM = args.trail_num
BATCH_SIZE = args.batch_size
EPOCH = args.epoch
DIS_DIM = args.disturbance_dim
DIS_MAX_RATIO = args.disturbance_max_ratio
DIS_STEP = args.disturbance_step
CAUSAL_GAT_LENGTH = args.causal_gat_length
CAUSAL_GAT_OUTPUT = args.causal_gat_output
CAUSAL_MAT_AGGRE = args.aggeragate_causal_matrix
VERB_FREQ = args.verbose_frequency
LR = args.lr
OPTIMIZER = args.optimizer
PATIENCE = args.patience
DOWN_MODEL = args.downstream_model

# Create the Files which we will use later.
if not args.trained_model:
    os.makedirs(f"Model/{DATA}/{TIME}", exist_ok=True)
    os.makedirs(f"log/{TIME}", exist_ok=True)

    # logger and writer
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    fh = logging.FileHandler(os.path.join(f"log/{TIME}", f"{DATA}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    writer = SummaryWriter(f"log/{TIME}/runs")


def logger_or_print(str, end="\n"):
    if not args.trained_model:
        logger.info(str)
    else:
        print(str, end=end)


accelerator = Accelerator()

# loading the dataset and dataloader


def loader_prepare(loader):
    dataloader = torch.load(f"Data/{DATA}/{DATA}_dataset_{loader}.pt").loader(
        BATCH_SIZE
    )
    dataloader.dataset.X_tensor = dataloader.dataset.X_tensor.to(DEVICE)
    dataloader.dataset.adj_tensor = dataloader.dataset.adj_tensor.to(DEVICE)
    dataloader.dataset.ts_count = dataloader.dataset.ts_count.to(DEVICE)
    dataloader.dataset.initial_weight = dataloader.dataset.initial_weight.to(DEVICE)
    dataloader = accelerator.prepare(dataloader)
    return dataloader


train_dataloader = loader_prepare("train")
valid_dataloader = loader_prepare("valid")

TYPE_NUM = train_dataloader.dataset.uer_item_type_num
TRAIN_SIZE = len(train_dataloader)

if not args.trained_model:

    class train:
        def __init__(self, model, optimizer, logger, writer, accelerator):
            self.model = model
            self.optimizer = optimizer
            self.logger = logger
            self.writer = writer
            self.accelerator = accelerator
            self.cum_causal_matrix = torch.where(self.model.GAT.causal_matrix.detach().clone() > 0, 1/2, 0.)

            self.PN_dict = {0: "positive", 1: "negative"}

        def single_train(self, dataloader, epoch, trail, step, eval_mode=False):
            """train for a single epoch

            Args:
                dataloader (DataLoader): the dataloader.
                epoch (int): the epoch now.
                trail (int): the trail now.
                step (float): update rate for the cum causal matrix.
                eval_mode (bool, optional): whether in the eval mode. Defaults to False.
            """
            print(self.cum_causal_matrix.sum())
            loss_sum = 0
            size = len(dataloader)
            Disturber = SVD_GFT(dataloader.dataset, DIS_DIM, DIS_MAX_RATIO)

            if not eval_mode:
                self.model.train()
                causal_matrix_change = self.cum_causal_matrix.unsqueeze(0)

                for batch, (in_group_index, ts_index, disturb_ts_index) in enumerate(
                    dataloader
                ):
                    causal_matrix = self.model.GAT.causal_matrix.detach().clone()
                    self.cum_causal_matrix = (
                        step * causal_matrix + (1 - step) * self.cum_causal_matrix
                    )
                    causal_sum = self.cum_causal_matrix + self.cum_causal_matrix.T
                    self.cum_causal_matrix[(causal_sum > 0) & (causal_sum < 1)] = 1/2
                    latest_causal_matrix = torch.where(
                        self.cum_causal_matrix > 0.5,
                        1.,
                        0,
                    )
                    causal_matrix_change = torch.cat(
                        [causal_matrix_change, latest_causal_matrix.unsqueeze(0)], dim=0
                    )
                    (
                        x,
                        x_disturbed_positive,
                        x_disturbed_negative,
                    ) = Disturber.disturbance(
                        in_group_index, ts_index, disturb_ts_index, latest_causal_matrix
                    )
                    adj = dataloader.dataset.adj_tensor[
                        ts_index
                    ]  # shape (group_num, N, N) -> (batch_size, N, N)
                    positive_similarity = sum(self.model(x, x_disturbed_positive, adj))
                    negative_similarity = sum(self.model(x, x_disturbed_negative, adj))
                    # loss = positive_similarity - negative_similarity
                    loss = -torch.log(
                        torch.sigmoid((positive_similarity - negative_similarity)/10)
                    )
                    loss_sum += loss.item()

                    self.writer.add_scalar(
                        f"Trail{trail+1}/train_batch_loss", loss, epoch * size + batch
                    )

                    # Backpropagation
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss, retain_graph=True)
                    self.optimizer.step()

                    if (batch % VERB_FREQ == 0) and batch:
                        try:
                            batch_loss_sum = loss_sum - last_loss_sum
                            last_loss_sum = loss_sum
                        except Exception:
                            last_loss_sum = 0
                            batch_loss_sum = loss_sum - last_loss_sum
                        current = batch + 1
                        self.logger.debug(
                            f"average train loss for the recent {VERB_FREQ} batches: {batch_loss_sum/VERB_FREQ:>7f}  [{current:>5d}/{size:>5d}]"
                        )

                return loss_sum / size, causal_matrix_change

            else:
                self.model.eval()
                loss_sum = 0
                causal_matrix = torch.where(
                        self.cum_causal_matrix > 0.5,
                        1.,
                        0,
                    )
                for batch, (in_group_index, ts_index, disturb_ts_index) in enumerate(
                    dataloader
                ):
                    (
                        x,
                        x_disturbed_positive,
                        x_disturbed_negative,
                    ) = Disturber.disturbance(
                        in_group_index,
                        ts_index,
                        disturb_ts_index,
                        causal_matrix,
                    )
                    adj = dataloader.dataset.adj_tensor[
                        ts_index
                    ]  # shape (group_num, N, N) -> (batch_size, N, N)
                    positive_similarity = sum(self.model(x, x_disturbed_positive, adj))
                    negative_similarity = sum(self.model(x, x_disturbed_negative, adj))
                    # loss = positive_similarity - negative_similarity
                    loss = -torch.log(
                        torch.sigmoid((positive_similarity - negative_similarity)/10)
                    )
                    loss_sum += loss.item()
                return loss_sum / size

        def start(self, epoch, step, trail):
            causal_matrix_change = torch.tensor([])
            best_loss = torch.inf
            count = 0
            for i in range(epoch):
                self.logger.info(
                    f"Epoch {i+1}, {TRAIN_SIZE} batches in total-------------------------------"
                )
                average_train_loss, batch_causal_matrix_change = self.single_train(
                    dataloader=train_dataloader, epoch=i, trail=trail, step=step
                )
                causal_matrix_change = torch.cat(
                    [causal_matrix_change, batch_causal_matrix_change.cpu()], dim=0
                )
                average_valid_loss = self.single_train(
                    dataloader=valid_dataloader,
                    epoch=i,
                    trail=trail,
                    step=step,
                    eval_mode=True,
                )
                self.writer.add_scalar(
                    f"Trail{trail+1}/train_epoch_loss", average_train_loss, i
                )
                self.writer.add_scalar(
                    f"Trail{trail+1}/valid_epoch_loss", average_valid_loss, i
                )

                if average_valid_loss < best_loss:
                    count = 0
                    best_loss = average_valid_loss
                    torch.save(
                        causal_matrix_change,
                        f"Model/{DATA}/{TIME}/{DATA}_trail_{trail+1}_causal.pt",
                    )
                    self.logger.info(
                        f"Epoch {i+1} finished, the average valid loss is {average_valid_loss}, \
                            \n and the model is saved"
                    )
                else:
                    count += 1
                    if count == PATIENCE:
                        self.logger.info(
                            f"Epoch {i+1} finished, the average valid loss is {average_valid_loss}, \
                                \n which is not better than the best loss {best_loss}, patience:{count}/{PATIENCE}, the training is stopped"
                        )
                        break
                    else:
                        self.logger.info(
                            f"Epoch {i+1} finished, the average valid loss is {average_valid_loss}, \
                                \n which is not better than the best loss {best_loss}, patience:{count}/{PATIENCE}"
                        )

            self.logger.info(
                f"Trail {trail+1} finished, the final average valid loss is {best_loss}"
            )

            return best_loss

    logger.info(
        f"The model parameters are as below: \n DATA = {DATA} \n EPOCH = {EPOCH}\n BATCH_SIZE = {BATCH_SIZE} \n LR = {LR} \n OPTIMIZER = {OPTIMIZER}\
            \n PATIENCE = {PATIENCE} \n CAUSAL_GAT_LENGTH = {CAUSAL_GAT_LENGTH} \n CAUSAL_GAT_OUTPUT = {CAUSAL_GAT_OUTPUT} \n CAUSAL_MAT_AGGRE = {CAUSAL_MAT_AGGRE}\
                \n DIS_STEP = {DIS_STEP} \n DIS_DIM = {DIS_DIM} \n DIS_MAX_RATIO = {DIS_MAX_RATIO} \n TRAIL_NUM = {TRAIL_NUM}"
    )
    logger.info(f"Start training for {TRAIL_NUM} trails")
    loss = []
    for i in range(TRAIL_NUM):
        path = f"Model/{DATA}/{TIME}/{DATA}_trail_{i+1}.pt"
        model = CausalGAT(
            dataset=train_dataloader.dataset,
            type_num=TYPE_NUM,
            length=CAUSAL_GAT_LENGTH,
            output=CAUSAL_GAT_OUTPUT,
            aggregate_method=CAUSAL_MAT_AGGRE,
        )
        optimizer = eval(f"torch.optim.{OPTIMIZER}(model.parameters(), lr={LR})")
        model, optimizer = accelerator.prepare(model, optimizer)
        Train = train(
            model=model,
            optimizer=optimizer,
            logger=logger,
            writer=writer,
            accelerator=accelerator,
        )
        Train = accelerator.prepare(Train)
        logger.info(f"Trail {i+1} start")
        loss.append(Train.start(EPOCH, DIS_STEP, trail=i))
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), path)
    logger.info(
        f"The average valid loss is {sum(loss)/len(loss)} \
            \n The standard deviation is {torch.std(torch.tensor(loss)).item()}"
    )
else:
    print("Start loading the model")
    model = CausalGAT(
        dataset=train_dataloader.dataset,
        type_num=TYPE_NUM,
        length=CAUSAL_GAT_LENGTH,
        output=CAUSAL_GAT_OUTPUT,
        aggregate_method=CAUSAL_MAT_AGGRE,
    )
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(torch.load(args.trained_model))
    model = accelerator.prepare(model)
    print("Model loaded")

if TRAIL_NUM == 1:
    # After training, generate the whole encoder and save them.

    def add_edge_feature(edge_feature, dataset):
        for i in range(dataset.ts_num):
            X = dataset.X_tensor[i][: dataset.ts_count[i]]
            adj = dataset.adj_tensor[i]
            for x in torch.split(X, BATCH_SIZE):
                edge_feature.append(model.map(x, adj).cpu())
        return edge_feature

    logger_or_print("Start generating the encoder")
    train_dataset = train_dataloader.dataset
    valid_dataset = valid_dataloader.dataset
    test_dataset = torch.load(f"Data/{DATA}/{DATA}_dataset_test.pt")
    test_dataset.X_tensor = test_dataset.X_tensor.to(DEVICE)
    test_dataset.adj_tensor = test_dataset.adj_tensor.to(DEVICE)
    edge_feature = []
    edge_feature = add_edge_feature(edge_feature, train_dataset)
    edge_feature = add_edge_feature(edge_feature, valid_dataset)
    edge_feature = add_edge_feature(edge_feature, test_dataset)
    edge_feature = torch.cat(
        edge_feature, dim=0
    )  # shape (N, length, 2), the last dimension is for the user and item embedding.
    node_feature = model.embedding.weight[
        :, -2 * CAUSAL_GAT_LENGTH:
    ]  # shape (N, 2*length)
    logger_or_print("Encoder generated")

    if DOWN_MODEL == "TGN":
        from numpy import save, pad

        logger_or_print("Start save the encoder for TGN")
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        OUT_FEAT = f"./TGN/data/ml_{DATA}.npy"
        edge_feature = edge_feature.detach().numpy().mean(axis=2)
        save(
            OUT_FEAT, pad(edge_feature, ((1, 0), (0, 0)), "constant", constant_values=0)
        )
        OUT_NODE_FEAT = f"./TGN/data/ml_{DATA}_node.npy"
        node_feature = node_feature.cpu().detach().numpy()
        save(
            OUT_NODE_FEAT,
            pad(node_feature, ((1, 0), (0, 0)), "constant", constant_values=0),
        )

        logger_or_print("Encoder saved")
