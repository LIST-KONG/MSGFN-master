import logging

from ppnp import PPNP
from training import train_model
from earlystopping import stopping_args
from propagation import PPRExact, PPRPowerIteration
from data.io import load_dataset, networkx_to_sparsegraph, load_mdd
import numpy as np
logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
graph_name = 'MDD'
if graph_name == 'MDD':
        graph = load_mdd()
else:
        graph = load_dataset(graph_name)
graph.standardize(select_lcc=True)
prop_ppnp = PPRExact(graph.adj_matrix, alpha=0.1)
prop_appnp = PPRPowerIteration(graph.adj_matrix, alpha=0.1, niter=10)
model_args = {
    'hiddenunits': [64],
    #之前64
    'drop_prob': 0.5,
    'propagation': prop_appnp}
idx_split_args = {'ntrain_per_class': 20, 'nstopping': 20, 'nknown': 30, 'seed': 2413340114}
reg_lambda = 5e-3
learning_rate = 0.05
#0.01

test = True
device = 'cuda'
print_interval = 20


# model, result = train_model(
        # graph_name, PPNP, graph, model_args, learning_rate, reg_lambda,
        # idx_split_args, stopping_args, test, device, None, print_interval)

if __name__ == '__main__':
    acclist = []
    senlist = []
    spelist = []
    for epoch in range(10):
        # np.random.seed(epoch*10)
        # torch.manual_seed(epoch*10)
        # if args.cuda:
        #     torch.cuda.manual_seed(epoch*10)
        # measures = train_gcn()
        model, result, acc, sen, spe = train_model(
        graph_name, PPNP, graph, model_args, learning_rate, reg_lambda,
        idx_split_args, stopping_args, test, device, None, print_interval)
        acclist.append(acc)
        senlist.append(sen)
        spelist.append(spe)

    accmean = np.mean(acclist)
    senmean = np.mean(senlist)
    spemean = np.mean(spelist)
    accmax = max(acclist, key = abs)
    accmaxi = acclist.index(accmax)
    spemax = spelist[accmaxi]
    senmax = senlist[accmaxi]
    accstd = np.std(acclist, ddof=1)
    spestd = np.std(spelist, ddof=1)
    senstd = np.std(senlist, ddof=1)
    print("all_acc={:.4f}, all_sen={:.4f}, all_spe={:.4f}".format(accmean, senmean, spemean))
    print(
        "acc={:.2f}+-{:.2f}  sen={:.2f}+-{:.2f}  spe={:.2f}+={:.2f}".format(accmean * 100, accstd * 100, senmean * 100,
                                                                            senstd * 100, spemean * 100, spestd * 100))
    print("best_acc={:.4f}, sen={:.4f}, spe={:.4f}".format(accmax, senmax, spemax))