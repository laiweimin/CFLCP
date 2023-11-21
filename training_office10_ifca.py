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

from image_helper_officeCaltech10 import ImageHelper
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

def train(helper, epoch, train_data_sets, local_model, target_model, LR=0.1,cluster=5
          ):
    ### Accumulate weights for all participants.
    weight_accumulator_list = [{} for _ in range(cluster)]
    cluster_list=[[] for _ in range(cluster)]

    for name, data in local_model.state_dict().items():
        #### don't scale tied weights:
        for i in range(cluster):
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator_list[i][name] = torch.zeros_like(data)

    for model_id in range(len(train_data_sets)):
        client_loss_list=[]
        client_weight_list=[]
        client_id, (current_data_model, train_data) = train_data_sets[model_id]
        helper.top_model.eval()
        for i in range(cluster):
            model = local_model
            model.copy_params(target_model[i].state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                        momentum=helper.params['momentum'],
                                        weight_decay=helper.params['decay'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
            model.train()
            total_loss = 0.
            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):  ### 选中的参与者，训练2次，每次都用全部数据
                data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch, evaluation=False)
                    pro1, _, output = model(data)
                    loss = nn.functional.cross_entropy(output, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data
            #记录当前客户端在所有簇模型的权重和损失大小
            client_weight_list.append(copy.deepcopy(model.state_dict()))
            client_loss_list.append(copy.deepcopy(total_loss))
        #选择最小损失的簇模型聚合
        client_cluster_id=client_loss_list.index(min(client_loss_list))
        cluster_list[client_cluster_id].append(client_id)
        for name in client_weight_list[client_cluster_id]:
            weight_accumulator_list[client_cluster_id][name] = weight_accumulator_list[client_cluster_id][name] + client_weight_list[client_cluster_id][name]
    print(cluster_list)
    for i in range(cluster):
        if len(cluster_list[i]) < 1:
            continue
        for key in weight_accumulator_list[i]:
            weight_accumulator_list[i][key] = weight_accumulator_list[i][key] * (1/len(cluster_list[i]))
    return weight_accumulator_list,cluster_list


def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    total_data_size=0
    for model_id in range(len(data_source)):
        data_iterator = data_source[model_id][1]
        total_data_size=total_data_size+len(data_iterator.sampler)
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_source, batch, evaluation=True)

            _, _, output = model(data)
            if len(batch)>1:
                total_loss += nn.functional.cross_entropy(output, targets,
                                                              reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()


    acc = 100.0 * (float(correct) / float(total_data_size))
    total_l = total_loss / total_data_size

    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                        'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                           total_l, correct, total_data_size,
                                                           acc))

    if visualize:
        model.visualize(vis, epoch, acc, total_l if helper.params['report_test_loss'] else None,
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

    helper.load_data(noniid=args.noniid)
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
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        t = time.time()
        start_time = time.time()
        gobal_weight_accumulator = dict()
        for name, data in helper.top_model.state_dict().items():
            gobal_weight_accumulator[name] = torch.zeros_like(data)

        t = time.time()
        LR = LR * 0.9995
        regions_num = 0
        select_set = []
        intra_agg_weight=[]
        subset_data_chunks = []
        subset_data_chunks = random.sample(range(20), 8)

        weight_accumulator_list, cluster_list = train(helper=helper, epoch=epoch,
                                                      train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                                       subset_data_chunks],
                                                      local_model=helper.local_model,
                                                      target_model=helper.model_list,
                                                      LR=LR, cluster=n_clusters
                                                      )
        for i in range(n_clusters):
            if len(cluster_list[i])<1:
                continue
            helper.model_list[i].load_state_dict(weight_accumulator_list[i])
        logger.info(f'Selected models: {subset_data_chunks},lr:{LR}')
        logger.info(f'time spent on training: {time.time() - t}')


        if epoch %1 ==0:
            regions = clu_test(helper=helper, cluster=n_clusters, train_data_sets=helper.train_data)
            # acc=0.0
            for i in range(len(regions)):
                if len(regions[i])<1:
                    continue
                epoch_loss, epoch_acc = test(helper=helper, epoch=epoch, data_source=[helper.test_data[x] for x in regions[i]],
                                             model=helper.model_list[i], is_poison=False, visualize=True)
                # acc=acc+epoch_acc/len(regions)
            # print(acc)


        if epoch % 1000 == 0:
            for i in range(4):
                # helper.save_model(model=eval('helper.target' + str(i) + '_model'),epoch=epoch, val_loss=epoch_loss)
                helper.save_model(model=helper.model_list[i], epoch=epoch, val_loss=epoch_loss)
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
