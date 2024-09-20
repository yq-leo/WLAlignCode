from collections import defaultdict

from network import *
import torch.optim as optim
from trainer.MyLoss import *
from trainer.MyOptimizer import adam
# from trainer.MyTrain4Struct import *
import torch.nn as nn
from utils.utils4Mapping import *
# from trainer.Trainer import *
from trainer.Trainer_T import *
from copy import copy
from utils.utils4ReadData import *
import os
import argparse
# from testEmb import testEmb
from testpAtN import test_performance, rm_out
import csv


def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Parameter setting
EMBEDDING_DIM = 128
PRINT_EVERY = 100
EPOCHS = 100
EPOCHS4Struct = 0
BATCH_SIZE = 1000
BATCHS = 100
N_SAMPLES = 20
LR = 0.005
LR4Struct = 0.005
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pe', help='dataset name.')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--ratio', type=int, default=2, help='training ratio')
parser.add_argument('--edge_noise', type=float, default=0.0, help='edge noise')
parser.add_argument('--attr_noise', type=float, default=0.0, help='attribute noise')
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--record', action='store_true', help='record results')
parser.add_argument('--robust', action='store_true', help='remove metric outliers')
args = parser.parse_args()

print(args)

data_file = args.dataset
ratio = args.ratio
# File parameters
ouput_filename_networkx, \
    ouput_filename_networky, \
    networkx_file, \
    networky_file, \
    anchor_file, \
    test_file = get_data(ratio, data_file)

edge_noise = args.edge_noise
attr_noise = args.attr_noise

top_k = [1, 5, 10, 30, 50, 100]
hits_list = defaultdict(list)
mrr_list = list()

for run in range(args.runs):
    print(f"Run {run + 1}/{args.runs}")
    # Read network and related structure data
    nx_G, G_f, G_t, anchor_list = read_graph(networkx_file, networky_file, anchor_file, edge_noise, attr_noise)
    graph_B = nx.DiGraph()

    G_anchor = get_graph_anchor(nx_G)

    network = networkC(G_anchor, nx_G, True, 1, 1, anchor_list)
    network_all = networkC(nx_G, G_anchor, True, 1, 1, anchor_list)

    init_dim = len(network.vocab2int)
    print(network.num_nodes())
    print('length of all nodes:', len(list(network.G.nodes())))
    print('length of all nodes:', len(list(network.G_all.nodes())))
    length_nodes = len(list(network.G_all.nodes()))
    test_anchor_list = get_test_anchor(test_file)

    print(device)
    all_candate = []
    un_candate_pair_list = []
    neg_result = []
    embedding_dict_list = []
    n = 100
    is_change = False
    all_mark = 0
    all_mis = 0
    all_closure = dict()

    # Initialize training function
    # trainerI = Trainer(EMBEDDING_DIM, 0.0001, EPOCHS4Struct, BATCHS, BATCH_SIZE, N_SAMPLES, network_all, device)
    trainerT = Trainer_T(EMBEDDING_DIM, LR4Struct, 8000, BATCHS, BATCH_SIZE, N_SAMPLES, network_all, device)

    y = 1
    num_mark = 0
    acc_list = []

    # Training process
    iter = 0
    while True:
        print('iter:', iter)
        print(len(network.vocab2int))
        # Initialize aggregate function and node label
        agg_model = AggregateLabel(len(network.vocab2int), len(network.vocab2int), device).to(device)
        onehot_label = get_graph_label(network, len(network.vocab2int), device).to(device)
        # if y==1:
        #     pre_layer_label=torch.zeros(len(network_all.vocab2int),EMBEDDING_DIM).to(device)
        # else:
        #     pre_layer_label=embedding_I.detach()
        # print(pre_layer_label.shape,'============================================')

        # label aggregate=================================================================
        layers_label = agg_model(onehot_label.weight.data, network.adj_real.to(device), 1)

        # Judge the colored node pair by similarity=================================================================
        candate_pair, un_candate_pair, result_old_list = get_candate_pair(layers_label[0], network)
        candate_pair_self, un_candate_pair_self, result_old_list_self = get_candate_pair_self(layers_label[0], network)

        y += 1

        # Judge whether convergence
        if not network.is_convergence():
            network_tmp = copy(network)
            network_tmp.vocab2int = copy(network.vocab2int)
            network_tmp.int2vocab = copy(network.int2vocab)
            candate_pair = list(set(candate_pair))
            all_candate.extend(candate_pair)
            all_candate = list(set(all_candate))

            # embedding_I, network_all = trainerI.train_anchor(network_tmp, all_candate, layers_label[0],pre_layer_label, nx_G,network.mark_pair,0)
            #
            embedding_T, network_all = trainerT.train_anchor(network_all, all_candate, un_candate_pair_list,
                                                             layers_label[0], network.mark_pair,
                                                             ouput_filename_networkx, ouput_filename_networky, 50)
            # If there is no convergence, remap the label
            network.remark_node(candate_pair)
            network.reset_anchor(candate_pair, get_graph_anchorBy_mark, len(candate_pair))

            print(network.is_mark_finished())
            network.reset_edges(candate_pair, get_graph_anchorBy_mark, candate_pair_self)
            num_mark = network.num_mark()
            iter += 1
            continue
        else:
            num_mark = network.num_mark()

            candate_pair = list(set(candate_pair))
            all_candate = remark(all_candate, network)
            # all_candate = list(set(all_candate))

            all_candate.extend(candate_pair)

            embedding_T, network_all = trainerT.train_anchor(network_all, all_candate, un_candate_pair_list,
                                                             layers_label[0], network.mark_pair,
                                                             ouput_filename_networkx, ouput_filename_networky)
            print(embedding_T.shape)
            writeFile(embedding_T, network_all, ouput_filename_networkx + ".number_T", "_foursquare")
            writeFile(embedding_T, network_all, ouput_filename_networky + ".number_T", "_twitter")

            break
    print(f'finished! number of iterations: {iter}')

    # Test performance
    hits, mrr = test_performance(data_file, ratio, edge_noise)
    for k in top_k:
        hits_list[k].append(hits[k])
    mrr_list.append(mrr)

hits_mean = dict()
hits_std = dict()
for k in top_k:
    if args.robust:
        hits_list[k] = rm_out(np.array(hits_list[k]))
    hits_mean[k] = np.mean(hits_list[k])
    hits_std[k] = np.std(hits_list[k])
mrr_mean = np.mean(mrr_list)
mrr_std = np.std(mrr_list)

if args.record:
    exp_name = "edge_noise"
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(f"results/{exp_name}_test.csv"):
        with open(f"results/{exp_name}_test.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([""] + [f"Hit@{k}" for k in top_k] + ["MRR"] + [f"std@{k}" for k in top_k] + ["std_MRR"])

    with open(f"results/{exp_name}_test.csv", "a", newline='') as f:
        writer = csv.writer(f)
        header = f"{args.dataset}_({args.edge_noise:.1f})"
        writer.writerow(
            [header] + [f"{p:.3f}" for p in hits_mean] + [f"{mrr_mean:.3f}"] + [f"{p:.3f}" for p in hits_std] + [
                f"{mrr_std:.3f}"])
