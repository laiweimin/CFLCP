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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from image_helper_digit5_bak import ImageHelper
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


def train(helper, epoch, train_data_sets, local_model, target_model, is_poison, last_weight_accumulator=None, LR=0.1,
          old_model=None, protos_list=None, protos_main=None):
    ### Accumulate weights for all participants.
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        #### don't scale tied weights:
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

    ### This is for calculating distances
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    # region_id = int(model_id / 10)
    for model_id in range(len(train_data_sets)):
        model = local_model
        ## Synchronize LR and models
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        # optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
        model.train()

        start_time = time.time()
        _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        ### For a 'poison_epoch' we perform single shot poisoning

        if current_data_model == -1:
            ### The participant got compromised and is out of the training.
            #  It will contribute to poisoning,
            continue

        helper.top_model.eval()
        # old_model.eval()
        cos = torch.nn.CosineSimilarity(dim=-1)

        for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):  ### 选中的参与者，训练2次，每次都用全部数据
            total_loss = 0.
            if helper.params['type'] == 'text':
                data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
            else:
                data_iterator = train_data
            for batch_id, batch in enumerate(data_iterator):
                optimizer.zero_grad()
                data, targets = helper.get_batch(train_data, batch, evaluation=False)
                pro1, _, output = model(data)
                loss = nn.functional.cross_entropy(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.data

                if helper.params["report_train_loss"] and batch % helper.params['log_interval'] == 0 and batch > 0:
                    cur_loss = total_loss.item() / helper.params['log_interval']
                    elapsed = time.time() - start_time
                    logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'
                                .format(model_id, epoch, internal_epoch,
                                        batch, train_data.size(0) // helper.params['bptt'],
                                        helper.params['lr'],
                                        elapsed * 1000 / helper.params['log_interval'],
                                        cur_loss,
                                        math.exp(cur_loss) if cur_loss < 30 else -1.))
                    total_loss = 0
                    start_time = time.time()
                # logger.info(f'model {model_id} distance: {helper.model_dist_norm(model, target_params_variables)}')

        for name, data in model.state_dict().items():
            #### don't scale tied weights:
            # if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
            #     continue
            # if weight_accumulator[name].dtype!=torch.float32:
            #     weight_accumulator[name]=weight_accumulator[name].float()
            # weight_accumulator[name].add_((data - target_model.state_dict()[name])/len(train_data_sets))
            # weight_accumulator[name]=torch.where(torch.isinf(weight_accumulator[name]), torch.full_like(weight_accumulator[name], -np.inf), weight_accumulator[name])
            # weight_accumulator[name]=torch.where(torch.isnan(weight_accumulator[name]), torch.full_like(weight_accumulator[name], -np.inf), weight_accumulator[name])
            # weight_accumulator[name].add_((helper.top_model.state_dict()[name] - target_model.state_dict()[name])*0.5)
            weight_accumulator[name] = weight_accumulator[name] + data / float(len(train_data_sets))
    # if current_data_model > 40 and current_data_model < 60:
    #     print(total_loss)
    return weight_accumulator


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



def clu(helper, epoch, train_data_sets, local_model, top_model, domain=-1):
    top_model.eval()
    if epoch == 1:
        # top_model.train()
        # protos_init=[]
        # for x in range(0,1):
        #     weight_accumulator = dict()
        #     for name, data in top_model.state_dict().items():
        #         weight_accumulator[name] = torch.zeros_like(data)
        #     for i in range(len(train_data_sets)):
        #         model = local_model
        #         model.copy_params(top_model.state_dict())
        #         optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
        #                                     momentum=helper.params['momentum'],
        #                                     weight_decay=helper.params['decay'])
        #         model.train()
        #         current_data_model, train_data= train_data_sets[i]
        #         for internal_epoch in range(0, 1):
        #             data_iterator = train_data
        #             for batch_id, batch in enumerate(data_iterator):
        #                 optimizer.zero_grad()
        #                 data, targets = helper.get_batch(train_data, batch,evaluation=False)
        #                 _, pro1, output = model(data)
        #                 if batch_id == 1:
        #                     proto = pro1.sum(0)
        #                 elif batch_id > 1:
        #                     proto += pro1.sum(0)
        #                 loss = nn.functional.cross_entropy(output, targets)
        #                 loss.backward()
        #                 optimizer.step()
        #         model.eval()
        #         protos_init.append(proto)
        #         for name, data in model.state_dict().items():
        #             weight_accumulator[name]=weight_accumulator[name]+data*(1.0/len(train_data_sets))
        #     top_model.load_state_dict(weight_accumulator)
        top_model.eval()
        protos_init = []
        for i in range(len(train_data_sets)):
            current_data_model, train_data = train_data_sets[i]
            data_iterator = train_data
            for batch_id, batch in enumerate(data_iterator):
                data, targets = helper.get_batch(train_data, batch, evaluation=False)
                with torch.no_grad():
                    pro1, _, output = top_model(data)
                if batch_id == 0:
                    proto = pro1.sum(0)
                else:
                    proto += pro1.sum(0)
            protos_init.append(proto)

        cos = torch.nn.CosineSimilarity(dim=-1)
        protos_init = torch.tensor([item.cpu().detach().numpy() for item in protos_init]).cpu()

        # pca = PCA(n_components=3)
        # protos_init_3d = pca.fit_transform(np.array(protos_init))

        # tsne = TSNE(n_components=2, perplexity=17, n_iter=300)
        # # 使用t-SNE对数据进行降维
        # protos_init_3d = tsne.fit_transform(protos_init)
        # colors_d = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
        # domains_d = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
        # for i in range(5):
        #     plt.scatter(protos_init_3d[20*i:20*i+20, 0], protos_init_3d[20*i:20*i+20, 1],c=colors_d[i], label=domains_d[i])
        # plt.title("Digits5 t-SNE Visualization")
        # plt.legend()
        # plt.savefig('scatter_plot.svg', format='svg')
        # plt.show()


        # similarity_matrix = np.zeros([len(protos_init), len(protos_init)])
        # for i in range(len(protos_init)):
        #     for j in range(len(protos_init)):
        #         similarity_matrix[i, j] = cos(protos_init[i], protos_init[j])
        # print(similarity_matrix)
        # dbmean_init = DBSCAN(eps=3).fit(protos_init)
        # print(dbmean_init.labels_)
        mean_init=KMeans(random_state=0, n_clusters=4).fit(protos_init)
        regions=[[]for i in range(mean_init.labels_.max()+1)]
        for i, label in enumerate(mean_init.labels_):
            regions[label].append(train_data_sets[i][0])
        print(regions)
        return regions
    else:
        protos_init = []
        for i in range(len(train_data_sets)):
            current_data_model, train_data = train_data_sets[i]
            batch_size = helper.params['batch_size']
            data_iterator = train_data
            for batch_id, batch in enumerate(data_iterator):
                data, targets = helper.get_batch(train_data, batch, evaluation=False)
                with torch.no_grad():
                    pro1, _, output = top_model(data)
                if batch_id == 0:
                    proto = pro1.sum(0)
                else:
                    proto += pro1.sum(0)
            protos_init.append(proto)

        cos = torch.nn.CosineSimilarity(dim=-1)
        client = [i for i in range(len(train_data_sets))]
        protos_init = torch.tensor([item.cpu().detach().numpy() for item in protos_init]).cpu()

        # pca = PCA(n_components=3)
        # protos_init_3d = pca.fit_transform(np.array(protos_init))

        # tsne = TSNE(n_components=2, perplexity=17, n_iter=300)
        # # 使用t-SNE对数据进行降维
        # protos_init_3d = tsne.fit_transform(protos_init)
        # colors_d = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']
        # domains_d = ['mnistm', 'mnist', 'syn', 'usps', 'svhn']
        # for i in range(5):
        #     plt.scatter(protos_init_3d[20*i:20*i+20, 0], protos_init_3d[20*i:20*i+20, 1],c=colors_d[i], label=domains_d[i])
        # plt.title("Digits5 t-SNE Visualization")
        # plt.legend()
        # plt.savefig('scatter_plot.svg', format='svg')
        # plt.show()

        similarity_matrix = np.zeros([len(protos_init), len(protos_init)])
        for i in range(len(protos_init)):
            for j in range(len(protos_init)):
                similarity_matrix[i, j] = cos(protos_init[i], protos_init[j])
        print(similarity_matrix)
        # dbscan = DBSCAN(eps=0.01, min_samples=2, metric='precomputed')  # 调整eps和min_samples以适应您的数据
        # cluster_labels = dbscan.fit_predict(similarity_matrix)
        # # 打印每个向量的簇分配
        # for i, label in enumerate(cluster_labels):
        #     print(f"Vector {i} belongs to Cluster {label}")

        # dbscan2 = DBSCAN(metric='precomputed')  # 调整eps和min_samples以适应您的数据
        # cluster_labels2 = dbscan2.fit_predict(similarity_matrix)
        # # 打印每个向量的簇分配
        # for i, label in enumerate(cluster_labels2):
        #     print(f"Vector {i} belongs to Cluster {label}")

        # similarity_log = np.log(similarity_matrix)
        # print(similarity_log)
        # dbscan3 = DBSCAN(metric='precomputed')  # 调整eps和min_samples以适应您的数据
        # cluster_labels3 = dbscan3.fit_predict(similarity_log)
        # # 打印每个向量的簇分配
        # for i, label in enumerate(cluster_labels3):
        #     print(f"Vector {i} belongs to Cluster {label}")

        # similarity_tanh = np.tanh(similarity_matrix)
        # print(similarity_tanh)
        # dbscan4 = DBSCAN(metric='precomputed')  # 调整eps和min_samples以适应您的数据
        # cluster_labels4 = dbscan4.fit_predict(similarity_tanh)
        # # 打印每个向量的簇分配
        # for i, label in enumerate(cluster_labels4):
        #     print(f"Vector {i} belongs to Cluster {label}")

        # print('a')
        # if (helper.params['sampling_dirichlet']):
        #     for i in range(10, 100):
        #         regions[cos(protos[:10], protos[i]).argmax().item()].append(i)
        # else:
        #     for i in range(1,100):
        #         if i%10==0:
        #             continue
        #         regions[cos(protos[:10], protos[i]).argmax().item()].append(i)
        regions = []
        # kmeans2 = KMeans(random_state=0).fit(protos2)
        # print(kmeans2)
        # # for i in range(10):
        # #     regions.append(np.where(kmeans2.labels_ == i)[0].tolist())
        # kmeans = KMeans(random_state=0).fit(protos)
        # kmeans_init = KMeans(random_state=0).fit(protos_init)

        # dbmean = DBSCAN(eps=0.5).fit(protos)
        # dbmean2 = DBSCAN(eps=0.5).fit(protos2)
        dbmean_init = DBSCAN(eps=5).fit(protos_init)

        # print(dbmean.labels_)
        # print(dbmean2.labels_)
        print(dbmean_init.labels_)
        # for i in range(10):
        #     regions.append(np.where(kmeans2.labels_ == i)[0].tolist())
        # while len(client)>0:
        #     dist2_cos=[]
        #     for i in client:
        #         dist2_cos.append(cos(protos2[client[0]], protos2[i]))
        #     p=[]
        #     for i in range(len(client)):
        #         if dist2_cos[i]>0.9:
        #             p.append(client[i])
        #     regions.append(p)
        #     for i in p:
        #         client.remove(i)
        # for i in range(10):
        #     print(regions[i])
        regions = [[] for i in range(dbmean_init.labels_.max() + 1)]
        # for i, label in enumerate(dbmean_init.labels_):
        #     regions[label].append(train_data_sets[i][0])
        # print(regions)
        return regions


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
    Dclient=20
    helper.load_data(noniid=args.noniid,Dclient=Dclient)
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

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()

    old_model_list = []
    region_proto = []
    protos_main = []

    for ii in range(10):
        helper.model_list[ii].copy_params(helper.top_model.state_dict())
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
        if args.noniid == 1:
            # feature shift
            subset_data_chunks = random.sample(range(100), 20)
        elif args.noniid == 2:
            # label shift
            subset_data_chunks = random.sample(range(Dclient * args.domain, Dclient * args.domain + Dclient), 8)
        elif args.noniid == 3:
            # feature and label shift
            subset_data_chunks = random.sample(range(100), 20)
        weight_accumulator = train(helper=helper, epoch=epoch,
                                   train_data_sets=[(pos, helper.train_data[pos]) for pos in
                                                    subset_data_chunks],
                                   local_model=helper.local_model,
                                   target_model=helper.top_model,
                                   is_poison=helper.params['is_poison'],
                                   last_weight_accumulator=weight_accumulator, LR=LR
                                   )
        helper.top_model.load_state_dict(weight_accumulator)
        logger.info(f'Selected models: {subset_data_chunks},lr:{LR}')
        logger.info(f'time spent on training: {time.time() - t}')

        if epoch %1 ==0:
            if args.noniid==2:
                epoch_loss, epoch_acc = test(helper=helper, epoch=epoch,
                                             data_source=[helper.test_data[x] for x in range(Dclient*args.domain, Dclient*args.domain+Dclient)],
                                             model=helper.top_model, is_poison=False, visualize=True)
            else:
                epoch_loss, epoch_acc = test(helper=helper, epoch=epoch,
                                             data_source=helper.test_data,
                                             model=helper.top_model, is_poison=False, visualize=True)

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
