import argparse
import copy
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.manifold import TSNE

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from image_helper_digit5_fedsoft import ImageHelper
from text_helper import TextHelper
from sklearn.cluster import KMeans, DBSCAN
from utils.utils import dict_html

logger = logging.getLogger("logger")
# logger.setLevel("ERROR")
import yaml
import time
import visdom
import numpy as np

vis = visdom.Visdom(port=8152)
import random
from utils.text_load import *

criterion = torch.nn.CrossEntropyLoss()

# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# random.seed(1)

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

def train(helper, epoch, train_data_sets, local_model, target_model, LR=0.1,cluster=5,importance_estimated=None
          ):
    ### Accumulate weights for all participants.
    weight_accumulator=dict()
    client_weight_list = []
    for name, data in local_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    for model_id in range(len(train_data_sets)):
        client_id, (current_data_model, train_data) = train_data_sets[model_id]
        model=helper.clients_model_list[client_id]
        model.train()
        # optimizer=torch.optim.Adam(params=model.parameters(), lr=LR)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):  ### 选中的参与者，训练2次，每次都用全部数据
            data_iterator = train_data
            for batch_id, batch in enumerate(data_iterator):
                data, targets = helper.get_batch(train_data, batch, evaluation=False)
                _, _, output = model(data)
                loss = nn.functional.cross_entropy(output, targets)
                mse_loss = nn.MSELoss(reduction='sum')
                for i, cluster in enumerate(helper.model_list[:n_clusters]):
                    l2 = None
                    for (param_local, param_cluster) in zip(model.parameters(), cluster.parameters()):
                        if l2 is None:
                            l2 = mse_loss(param_local, param_cluster)
                        else:
                            l2 += mse_loss(param_local, param_cluster)
                    loss += 0.01 / 2 * importance_estimated[model_id][i] * l2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()
        client_weight_list.append(copy.deepcopy(model.state_dict()))
    for key in weight_accumulator:
        for i in range(len(client_weight_list)):
            weight_accumulator[key] = weight_accumulator[key] + (client_weight_list[i][key]*(1/float(len(client_weight_list))))
    return weight_accumulator


def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):

    total_loss = 0
    correct = 0
    total_data_size=0
    for model_id in range(len(data_source)):
        model[model_id].eval()
        data_iterator = data_source[model_id][1]
        total_data_size=total_data_size+len(data_iterator.sampler)
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)

            _, _, output = model[model_id](data)
            if len(batch)>1:
                total_loss += nn.functional.cross_entropy(output, targets,
                                                              reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


    acc = 100.0 * (float(correct) / float(total_data_size))
    total_l = total_loss / total_data_size

    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model[model_id].name, is_poison, epoch,
                                                           total_l, correct, total_data_size,
                                                           acc))

    if visualize:
        model[model_id].visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
                        eid=helper.params['environment_name'], is_poisoned=is_poison)
    return (total_l, acc)

def clu_test(helper,cluster=5,train_data_sets=None):
    cluster_list = [[] for _ in range(cluster)]
    for j in range(cluster):
        helper.model_list[j].eval()
    for i in range(len(train_data_sets)):
        client_loss_list = []
        for j in range(cluster):
            loss=.0
            for batch_id, batch in enumerate(train_data_sets[i][1]):
                data, targets = helper.get_batch(train_data_sets[i][1], batch, evaluation=True)
                _, _, output = helper.model_list[j](data)
                if len(batch) > 1:
                    loss += nn.functional.cross_entropy(output, targets,reduction='sum').item()
            client_loss_list.append(copy.deepcopy(loss))
        client_cluster_id = client_loss_list.index(min(client_loss_list))
        cluster_list[client_cluster_id].append(i)
    print(cluster_list)
    return cluster_list

def estimate_importance_weights(helper,cluster=5,train_data_sets=None):
    with torch.no_grad():
        for j in range(cluster):
            helper.model_list[j].eval()
        all_importance_estimated=[]
        for i in range(len(train_data_sets)):
            cluster_list = [[] for _ in range(cluster)]
            nst_cluster_sample_count = [0] * cluster
            for j in range(cluster):
                for batch_id, batch in enumerate(train_data_sets[i][1]):
                    data, targets = helper.get_batch(train_data_sets[i][1], batch, evaluation=True)
                    _, _, output = helper.model_list[j](data)
                    loss=nn.functional.cross_entropy(output, targets,reduction='none')
                    cluster_list[j].extend(loss.cpu())
            min_loss_idx = np.argmin(np.array(cluster_list), axis=0)
            for s in range(cluster):
                nst_cluster_sample_count[s] += np.sum(min_loss_idx == s)
            for s in range(cluster):
                if nst_cluster_sample_count[s] == 0:
                    nst_cluster_sample_count[s] = 2
            importance_estimated = np.array([1.0 * nst / len(train_data_sets[i][1].sampler) for nst in nst_cluster_sample_count])
            all_importance_estimated.append(importance_estimated)
    return all_importance_estimated


if __name__ == '__main__':
    print('Start training')
    time_start_load_everything = time.time()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params')
    parser.add_argument('--sigma',type=float,default=0.8)
    parser.add_argument('--domain',type=int,default=-1)
    parser.add_argument('--noniid',type=int,default=1)
    args = parser.parse_args()
    print(args.sigma)
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    base_model = params_loaded['base_model']
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'),
                             dataname=1, single=False, divide_q=params_loaded['divide_q'], base_model=base_model)
    else:
        helper = TextHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'text'))

    Dclient = 20
    helper.load_data(noniid=args.noniid, Dclient=Dclient)
    helper.create_model(single=False)

    best_loss = float('inf')
    vis.text(text=dict_html(helper.params, current_time=helper.params["current_time"]),
             env=helper.params['environment_name'], opts=dict(width=300, height=400))
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))
    mean_acc = list()
    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None
    weight_accumulator_list = None
    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)


    n_clusters=4
    t = time.time()

    LR = helper.params['lr']
    if args.noniid == 1:
        all_importance_estimated = estimate_importance_weights(helper=helper, cluster=n_clusters,
                                        train_data_sets=helper.train_data)
    elif args.noniid==2:
        all_importance_estimated=estimate_importance_weights(helper=helper, cluster=n_clusters,train_data_sets=[helper.train_data[x] for x in range(Dclient*args.domain, Dclient*args.domain+Dclient)])
    else:
        all_importance_estimated = estimate_importance_weights(helper=helper, cluster=n_clusters,
                                                               train_data_sets=helper.train_data)
    importance_weights_matrix = np.array(all_importance_estimated)
    importance_weights_matrix /= np.sum(all_importance_estimated, axis=0)

    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        select_set = []
        for s in range(n_clusters):
            if args.noniid == 1:
                select_set.append(np.random.choice(a=range(100), size=4,
                                                   p=importance_weights_matrix[:, s], replace=False).tolist())
            elif args.noniid == 2:
                select_set.append(np.random.choice(a=range(Dclient*args.domain, Dclient*args.domain+Dclient), size=2,
                                                  p=importance_weights_matrix[:, s], replace=False).tolist())
            else:
                select_set.append(np.random.choice(a=range(100), size=4,
                                                   p=importance_weights_matrix[:, s], replace=False).tolist())
        print(select_set)
        t = time.time()
        start_time = time.time()
        gobal_weight_accumulator = dict()
        for name, data in helper.top_model.state_dict().items():
            gobal_weight_accumulator[name] = torch.zeros_like(data)

        t = time.time()
        LR = LR * 0.9995
        for i in range(n_clusters):
            subset_data_chunks=select_set[i]
            weight_accumulator= train(helper=helper, epoch=epoch,
                                                          train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                                           subset_data_chunks],
                                                          local_model=helper.local_model,
                                                          target_model=helper.model_list,
                                                          LR=LR, cluster=n_clusters,
                                      importance_estimated=[all_importance_estimated[pos%20] for pos in subset_data_chunks]
                                                          )
            helper.model_list[i].load_state_dict(weight_accumulator)
        logger.info(f'Selected models: {select_set},lr:{LR}')
        logger.info(f'time spent on training: {time.time() - t}')


        if epoch %1 ==0:
            # acc=0.0
            if args.noniid == 1:
                epoch_loss, epoch_acc = test(helper=helper, epoch=epoch,
                                             data_source=helper.test_data,
                                             model=helper.clients_model_list,
                                             is_poison=False, visualize=True)
            elif args.noniid==2:
                epoch_loss, epoch_acc = test(helper=helper, epoch=epoch,
                                             data_source=[helper.test_data[x] for x in range(Dclient*args.domain, Dclient*args.domain+Dclient)],
                                             model=[helper.clients_model_list[x] for x in range(Dclient*args.domain, Dclient*args.domain+Dclient)],
                                             is_poison=False, visualize=True)
            else:
                epoch_loss, epoch_acc = test(helper=helper, epoch=epoch,
                                             data_source=helper.test_data,
                                             model=helper.clients_model_list,
                                             is_poison=False, visualize=True)
                # acc=acc+epoch_acc/len(regions)
            # print(acc)


        if epoch % 1000 == 0:
            for i in range(5):
                # helper.save_model(model=eval('helper.target' + str(i) + '_model'),epoch=epoch, val_loss=epoch_loss)
                helper.save_model(model=helper.model_list[i], epoch=epoch, val_loss=epoch_loss)
            helper.save_model(model=helper.top_model, epoch=epoch, val_loss=epoch_loss)
        logger.info(f'Done in {time.time() - start_time} sec.')

    if helper.params['is_poison']:
        logger.info(f'MEAN_ACCURACY: {np.mean(mean_acc)}')
    logger.info('Saving all the graphs.')
    logger.info(f"This run has a label: {helper.params['current_time']}. "
                f"Visdom environment: {helper.params['environment_name']}")

    if helper.params.get('results_json', False):
        with open(helper.params['results_json'], 'a') as f:
            if len(mean_acc):
                results['mean_poison'] = np.mean(mean_acc)
            f.write(json.dumps(results) + '\n')

    vis.save([helper.params['environment_name']])
