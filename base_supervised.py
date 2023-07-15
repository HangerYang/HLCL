from HLCL.utils import seed_everything
import torch
import argparse
from HLCL.models import GCN

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = "cora", help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs' ,type=int, default = 500, help='pretraining epoch')
    parser.add_argument('--lr',type=float, default = 0.01, help='pre training learning rate')
    parser.add_argument('--runs', type=int, default=3, help='number of distinct runs')
    parser.add_argument('--neg', type=str, default="full_neg", help='number of distinct runs')
    parser.add_argument('--num_layer', type=int, default=2, help='number of layers')
    parser.add_argument('--device', type=int, default=1)
    args, unknown = parser.parse_known_args()
    return args

def get_split_given(data, run):
    split = {}
    split["train"] = torch.where(data.train_mask.t()[run])[0]
    split["valid"] = torch.where(data.val_mask.t()[run])[0]
    split["test"] = torch.where(data.test_mask.t()[run])[0]
    return split

args = get_arguments()
seed_everything(args.seed)
hidden_dim=512
device = args.device
data = dataset_split(dataset_name = args.dataset)
total_result = []
device = torch.device("cuda:{}".format(device))
data = data.to(device)
lr = args.lr
epochs = args.epochs
epoch = 0