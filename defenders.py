import torch
import numpy as np
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hdbscan
import torch.optim as optim
import imageio
import torch.nn as nn
from math import sqrt,log
from sklearn.decomposition import PCA
from collections import Counter
import time
from torch.nn.utils import vector_to_parameters, parameters_to_vector

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()


def load_model_weight_diff(net, weight_diff, global_weight):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    listed_global_weight = list(global_weight.parameters())
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight_diff[index_bias:index_bias + p.numel()].view(p.size()) + listed_global_weight[p_index]
        index_bias += p.numel()


class Defense:
    def __init__(self, *args, **kwargs):
        self.hyper_params = None

    def exec(self, client_model, *args, **kwargs):
        raise NotImplementedError()


class WeightDiffClippingDefense(Defense):
    def __init__(self, norm_bound, *args, **kwargs):
        self.norm_bound = norm_bound

    def exec(self, client_model, global_model, *args, **kwargs):
        """
        global_model: the global model at iteration T, bcast from the PS
        client_model: starting from `global_model`, the model on the clients after local retraining
        """
        vectorized_client_net = vectorize_net(client_model)
        vectorized_global_net = vectorize_net(global_model)
        vectorized_diff = vectorized_client_net - vectorized_global_net

        weight_diff_norm = torch.norm(vectorized_diff).item()
        clipped_weight_diff = vectorized_diff / max(1, weight_diff_norm / self.norm_bound)

        print("The Norm of Weight Difference between received global model and updated client model: {}".format(weight_diff_norm))
        print("The Norm of weight (updated part) after clipping: {}".format(torch.norm(clipped_weight_diff).item()))
        load_model_weight_diff(client_model, clipped_weight_diff, global_model)
        return None

class RSA(Defense):
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, global_model, flround, *args, **kwargs):

        for net_index, net in enumerate(client_model):
            whole_aggregator = []
            for p_index, p in enumerate(client_model[0].parameters()):
                params_aggregator = 0.00005 * 0.998 ** flround * torch.sign(list(net.parameters())[p_index].data
                    - list(global_model.parameters())[p_index].data) + list(global_model.parameters())[p_index].data

                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(net.parameters()):
                p.data = whole_aggregator[param_index]

        return None



class Krum(Defense):
    """
    we implement the robust aggregator at: https://papers.nips.cc/paper/6617-machine-learning-with-adversaries-byzantine-tolerant-gradient-descent.pdf
    and we integrate both krum and multi-krum in this single class
    """

    def __init__(self, mode, num_workers, num_adv, *args, **kwargs):
        assert (mode in ("krum", "multi-krum"))
        self._mode = mode
        self.num_workers = num_workers
        self.s = num_adv

    def exec(self, client_models, global_model_pre, num_dps, g_user_indices, device, *args, **kwargs):

        ######################################################################## separate model to get updated part
        whole_aggregator = []
        client_models_copy = copy.deepcopy(client_models)
        for i in range(len(client_models_copy)):
            for p_index, p in enumerate(client_models_copy[i].parameters()):
                params_aggregator = torch.zeros(p.size()).to(device)
                params_aggregator = params_aggregator + (list(client_models_copy[i].parameters())[p_index].data -
                                                         list(global_model_pre.parameters())[p_index].data)
                # params_aggregator = torch.sign(params_aggregator)
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(client_models_copy[i].parameters()):
                p.data = whole_aggregator[param_index]

            whole_aggregator = []

        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

        neighbor_distances = []
        for i, g_i in enumerate(vectorize_nets):
            distance = []
            for j in range(i + 1, len(vectorize_nets)):
                if i != j:
                    g_j = vectorize_nets[j]
                    distance.append(float(np.linalg.norm(g_i - g_j) ** 2))
            neighbor_distances.append(distance)

        # compute scores
        nb_in_score = self.num_workers - self.s - 2
        scores = []
        for i, g_i in enumerate(vectorize_nets):
            dists = []
            for j, g_j in enumerate(vectorize_nets):
                if j == i:
                    continue
                if j < i:
                    dists.append(neighbor_distances[j][i - j - 1])
                else:
                    dists.append(neighbor_distances[i][j - i - 1])

            topk_ind = np.argpartition(dists, nb_in_score)[:nb_in_score]
            scores.append(sum(np.take(dists, topk_ind)))
        if self._mode == "krum":
            i_star = scores.index(min(scores))
            print("===starr===:", i_star)
            print("===scoree===:", scores)
            print("@@@@ The chosen one is user: {}, which is global user: {}".format(scores.index(min(scores)),
                                                                                           g_user_indices[scores.index(
                                                                                               min(scores))]))
            aggregated_model = client_models[i_star]
            neo_net_list = [aggregated_model]
            print("Norm of Aggregated Model: {}".format(
                torch.norm(torch.nn.utils.parameters_to_vector(aggregated_model.parameters())).item()))
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, i_star
        elif self._mode == "multi-krum":
            topk_ind = np.argpartition(scores, nb_in_score + 2)[:nb_in_score + 2]

            # we reconstruct the weighted averaging here:
            selected_num_dps = np.array(num_dps)[topk_ind]

            print("===scores===", scores)
            print("Num data points: {}".format(num_dps))
            print("Num selected data points: {}".format(selected_num_dps))
            print("The chosen ones are users: {}, which are global users: {}".format(topk_ind,
                                                                    [g_user_indices[ti] for ti in topk_ind]))

            aggregated_model=[]
            for i in range(len(topk_ind)):
                aggregated_model.append(client_models[topk_ind[i]])

            neo_net_list = aggregated_model
            neo_net_freq = [1.0]
            return neo_net_list, neo_net_freq, topk_ind




class RFA(Defense):
    """
    we implement the robust aggregator at:
    https://arxiv.org/pdf/1912.13445.pdf
    the code is translated from the TensorFlow implementation:
    https://github.com/krishnap25/RFA/blob/01ec26e65f13f46caf1391082aa76efcdb69a7a8/models/model.py#L264-L298
    """

    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, maxiter=4, eps=1e-5, ftol=1e-6, device=torch.device("cuda"), *args, **kwargs):
        """
        Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        net_freq = [0.1 for i in range(len(client_model))]
        alphas = np.asarray(net_freq, dtype=np.float32)
        vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_model]
        median = self.weighted_average_oracle(vectorize_nets, alphas)

        num_oracle_calls = 1

        # logging
        obj_val = self.geometric_median_objective(median=median, points=vectorize_nets, alphas=alphas)

        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append("Tracking log entry: {}".format(log_entry))
        print('Starting Weiszfeld algorithm')
        print(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, self.l2dist(median, p)) for alpha, p in zip(alphas, vectorize_nets)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = self.weighted_average_oracle(vectorize_nets, weights)
            num_oracle_calls += 1
            obj_val = self.geometric_median_objective(median, vectorize_nets, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         self.l2dist(median, prev_median)]
            logs.append(log_entry)
            logs.append("Tracking log entry: {}".format(log_entry))
            print("#### Oracle Cals: {}, Objective Val: {}".format(num_oracle_calls, obj_val))
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        # print("Num Oracale Calls: {}, Logs: {}".format(num_oracle_calls, logs))

        aggregated_model = client_model[0]  # slicing which doesn't really matter
        load_model_weight(aggregated_model, torch.from_numpy(median.astype(np.float32)).to(device))
        neo_net_list = [aggregated_model]
        neo_net_freq = [1.0]
        return neo_net_list

    def weighted_average_oracle(self, points, weights):
        """Computes weighted average of atoms with specified weights
        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        ### original implementation in TFF
        # tot_weights = np.sum(weights)
        # weighted_updates = [np.zeros_like(v) for v in points[0]]
        # for w, p in zip(weights, points):
        #    for j, weighted_val in enumerate(weighted_updates):
        #        weighted_val += (w / tot_weights) * p[j]
        # return weighted_updates
        ####
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros(points[0].shape)
        for w, p in zip(weights, points):
            weighted_updates += (w * p / tot_weights)
        return weighted_updates

    def l2dist(self, p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # this is a helper function
        return np.linalg.norm(p1 - p2)

    def geometric_median_objective(self, median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * self.l2dist(median, p) for alpha, p in zip(alphas, points)])
class fltrust(Defense):
    def __init__(self, *args, **kwargs):
        pass

    def vectorize_net(self, net):
        return torch.cat([p.view(-1) for p in net.parameters()])

    def train(self, model, data_loader, device, criterion, optimizer):

        model.train()
        for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            output = model(batch_x)  # get predict label of batch_x
            loss = criterion(output, batch_y)  # cross entropy loss
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("loss: {}".format(loss))
        return model

    def exec(self, net_list, global_model, root_data, flr, lr, gamma, net_num, device, *args, **kwargs):

        root_net = copy.deepcopy(global_model)
        criterion = nn.CrossEntropyLoss()

        ############## training a root net using root dataset
        for e in range(1, 3):   ###  server side local training epoch could be adjusted
            optimizer = optim.SGD(root_net.parameters(), lr=lr * gamma ** (flr - 1),
                                  momentum=0.9,
                                  weight_decay=1e-4)  # epoch, net, train_loader, optimizer, criterion
            for param_group in optimizer.param_groups:
                print("Effective lr in fl round: {} is {}".format(flr, param_group['lr']))

            self.train(root_net, root_data, device, criterion, optimizer)

        root_update = copy.deepcopy(global_model)

        ##############  get  root_update
        whole_aggregator = []
        for p_index, p in enumerate(global_model.parameters()):
            params_aggregator = list(root_net.parameters())[p_index].data - list(global_model.parameters())[p_index].data
            whole_aggregator.append(params_aggregator)

        for param_index, p in enumerate(root_update.parameters()):
            p.data = whole_aggregator[param_index]

        ############## get user nets updates
        for i in range(net_num):
            whole_aggregator = []
            for p_index, p in enumerate(global_model.parameters()):
                params_aggregator = list(net_list[i].parameters())[p_index].data - list(global_model.parameters())[p_index].data
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(net_list[i].parameters()):
                p.data = whole_aggregator[param_index]


        ############# compute TS for all users
        root_update_vec = self.vectorize_net(root_update)
        TS = []
        net_vec_list = []
        for i in range(net_num):
            net_vec = self.vectorize_net(net_list[i])
            net_vec_list.append(net_vec)
            cos_sim = torch.cosine_similarity(net_vec, root_update_vec, dim=0)
            ts = torch.relu(cos_sim)
            TS.append(ts)
        if torch.sum(torch.Tensor(TS))==0:
            return global_model
        ############ get the regularized users' updates by aligning with root update
        norm_list = []
        for i in range(net_num):
            norm = torch.norm(root_update_vec) / torch.norm(net_vec_list[i])
            norm_list.append(norm)

        for i in range(net_num):
            whole_aggregator = []
            for p_index, p in enumerate(global_model.parameters()):
                params_aggregator = norm_list[i] * list(net_list[i].parameters())[p_index].data
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(net_list[i].parameters()):
                p.data = whole_aggregator[param_index]


        ########### aggregation: get global update
        whole_aggregator = []
        global_update = copy.deepcopy(global_model)

        for p_index, p in enumerate(net_list[0].parameters()):
            params_aggregator = torch.zeros(p.size()).to(device)
            for net_index, net in enumerate(net_list):
                params_aggregator = params_aggregator + TS[net_index] * list(net.parameters())[p_index].data
            whole_aggregator.append(params_aggregator)



        for param_index, p in enumerate(global_update.parameters()):
            p.data = (1/torch.sum(torch.tensor(TS))) * whole_aggregator[param_index]

        ########## get global model
        global_model_ = copy.deepcopy(global_model)
        for i in range(net_num):
            whole_aggregator = []
            for p_index, p in enumerate(global_model.parameters()):
                params_aggregator = list(global_update.parameters())[p_index].data + list(global_model.parameters())[p_index].data
                whole_aggregator.append(params_aggregator)

            for param_index, p in enumerate(global_model_.parameters()):
                p.data = whole_aggregator[param_index]

        return global_model_


  

class layering(Defense):
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, client_model, global_model, cut, device,*args, **kwargs):
        net_avg= copy.deepcopy(client_model[0])

        whole_aggregator = []
        for p_index, p in enumerate(client_model[0].parameters()):
            # initial
            params_aggregator = torch.zeros(p.size()).to(device)
            for net_index, net in enumerate(client_model):
                params_aggregator = params_aggregator + 1 / len(client_model) * list(net.parameters())[p_index].data

            whole_aggregator.append(params_aggregator)

        for i in range(len(client_model)):
            for param_index,p in enumerate(client_model[i].parameters()):
                if param_index>cut :
                    break

                p.data = whole_aggregator[param_index]

        for param_index, p in enumerate(net_avg.parameters()):
            p.data = whole_aggregator[param_index]

        return net_avg,client_model


class flame(Defense):
    def __init__(self, *args, **kwargs):
        pass

    def exec(self, global_model_pre, client_model, device, *args, **kwargs):

        net_avg = copy.deepcopy(global_model_pre)
        epsilon = 3705
        cos = []
        cos_ = []

        for i in range(len(client_model)):
            for j in range(len(client_model)):
                x1 = vectorize_net(client_model[i])-vectorize_net(net_avg)
                x2 = vectorize_net(client_model[j])-vectorize_net(net_avg)
                cos.append(torch.cosine_similarity((x1), (x2), dim=0).detach().cpu())

            cos_.append(torch.cat([p.view(-1) for p in cos]).reshape(-1, 1))

            cos = []

        cos_ = torch.cat([p.view(1, -1) for p in cos_])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        cluster_labels = clusterer.fit_predict(cos_)
        majority = Counter(cluster_labels)
        res = majority.most_common(len(client_model))

        out = []

        for i in range(len(cluster_labels)):
            if cluster_labels[i] == res[0][0]:
                out.append(i)

        e = []
        for i in range(len(client_model)):
            e.append(torch.sqrt(torch.sum((vectorize_net(net_avg) - vectorize_net(client_model[i])) ** 2)))


        e = torch.cat([p.view(-1) for p in e])

        st = torch.median(e)

        whole_aggregator = []
        par = []
        for i in range(len(out)):
            par.append(min(1, st / e[out[i]]))

        wa=[]
        for p_index, p in enumerate(net_avg.parameters()):
            wa.append(p.data)
        for p_index, p in enumerate(client_model[0].parameters()):
            # initial
            params_aggregator = torch.zeros(p.size()).to(device)

            for i in range(len(out)):
                net = client_model[out[i]]
                params_aggregator = params_aggregator + wa[p_index] + (list(net.parameters())[p_index].data-wa[p_index]) * par[i]
            sum = 0
            for i in range(len(par)):

                sum += 1
            params_aggregator = params_aggregator / sum
            whole_aggregator.append(params_aggregator)
        lamda=1e-3
        sigma = st * lamda

        for param_index, p in enumerate(net_avg.parameters()):
            p.data = whole_aggregator[param_index] + (sigma ** 2) * torch.randn(whole_aggregator[param_index].shape).to(device)

        return net_avg
