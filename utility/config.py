import argparse
import json

def get_configs(args):
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        raise Exception('config file not defined')
    if args.dataset is None:
        args.dataset = config['dataset']
    if args.epochs is None:
        args.epochs = config['epochs']
    if args.gnn is None:
        args.gnn = config['gnn']
    if args.preepochs is None:
            args.preepochs = config['preepochs']
    if args.loss_type is None:
            args.loss_type = config['loss_type']
    if args.seed is None:
        args.seed = config['seed']
    if args.pre_learning_rate is None:
        args.pre_learning_rate = config['pre_learning_rate']
    if args.learning_rate is None:
        args.learning_rate = config['learning_rate']
    if args.weight_decay is None:
        args.weight_decay = config['weight_decay']
    if args.patience is None:
        args.patience = config['patience']
    if args.hidden_dim is None:
        args.hidden_dim = config['hidden_dim']


    if args.aug_type is None: #adding more aug
        args.aug_type = config['aug_type']
    return args

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration file')
    parser.add_argument('--dataset', type=str, help='benchmark dataset : cora, citeseer, pubmed')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--preepochs', type=int, help='Number of epochs to pre-train, only applicable to FBGCN')
    parser.add_argument('--gnn', type=str, help='gcn, gat, fbgcn')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--patience', type=int, help='patience for early stopping')
    parser.add_argument('--aug_type', help='augmentation type')
    parser.add_argument('--aug_side', help='augmentation side')
    parser.add_argument('--hidden_dim' ,type=int, help='hidden dimension in the model')
    parser.add_argument('--second_hidden_dim' ,type=int, help='hidden dimension in the model')
    parser.add_argument('--pre_learning_rate',type=float, help='pre training learning rate')
    parser.add_argument('--loss_type', help='applying which loss')
    parser.add_argument('--aug', type=float, default = 0.3, help='aug parameter')
    parser.add_argument('--aug2', type=float, default = 0.3, help='aug parameter')
    parser.add_argument('--train_batch', type=str, default='cluster', help='type of mini batch loading scheme for training GNN')
    parser.add_argument('--no_mini_batch_test', action='store_true', help='whether to test on mini batches as well')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_parts', type=int, default=100, help='number of partitions for partition batching')
    parser.add_argument('--cluster_batch_size', type=int, default=1, help='number of clusters to use per cluster-gcn step')
    parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args = get_configs(args)
    return args


    