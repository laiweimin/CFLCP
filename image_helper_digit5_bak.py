import copy
from collections import defaultdict
from  generate_Digit5 import generate_Digit5
from  generate_Digit5 import generate_shift_Digit5
import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np
from utils import cifar
from models.resnet import ResNet18
from models.simple import SimpleMnist,Digit5CNN
from models.word_model import RNNModel
from utils.text_load import *
from utils.utils import SubsetSampler
from torch import nn
logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 2

device="cuda"

class ImageHelper(Helper):


    def poison(self):
        return

    def create_model(self,single=False):
        top_model = Digit5CNN(num_input=3,num_classes=10,name='Top',created_time=self.params['current_time'])
        local_model = Digit5CNN(num_input=3,num_classes=10,name='Local',created_time=self.params['current_time'])
        target_model = Digit5CNN(num_input=3,num_classes=10,name='Target',created_time=self.params['current_time'])

        local_model.to(device)
        target_model.to(device)
        top_model.to(device)

        model_list=[]
        for i in range(10):
            model_p = Digit5CNN(num_input=3,num_classes=10, name='Target'+str(i),
                                        created_time=self.params['current_time'])
            model_p.to(device)
            model_list.append(copy.deepcopy(model_p))
        self.model_list = model_list
        # if self.params['resumed_model']:
        #     if single:
        #         loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
        #         top_model.load_state_dict(loaded_params['state_dict'])
        #     else:
        #         s = self.params['resumed_model']
        #         ss = s[:s.find('Target')] + 'Top' + s[s.find('Target') + 7:]
        #         loaded_params = torch.load(f"saved_models/{ss}")
        #         top_model.load_state_dict(loaded_params['state_dict'])
        #         for i in range(47):
        #             s=self.params['resumed_model']
        #             ss=s[:s.find('Target')] + 'Target' + str(i) + s[s.find('Target') + 7:]
        #             loaded_params=torch.load(f"saved_models/{ss}")
        #             eval('target' + str(i) + '_model').load_state_dict(loaded_params['state_dict'])
        #     # loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
        #     # target_model.load_state_dict(loaded_params['state_dict'])
        #     self.start_epoch = loaded_params['epoch']
        #     self.params['lr'] = loaded_params.get('lr', self.params['lr'])
        #     logger.info(f"Loaded parameters from saved model: LR is"
        #                 f" {self.params['lr']} and current epoch is {self.start_epoch}")
        # else:
        #     self.start_epoch = 1

        self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
        self.top_model = top_model

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def poison_dataset(self):
        #
        # return [(self.train_dataset[self.params['poison_image_id']][0],
        # torch.IntTensor(self.params['poison_label_swap']))]
        cifar_classes = {}

        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        indices = list()
        # create array that starts with poisoned images

        #create candidates:
        # range_no_id = cifar_classes[1]
        # range_no_id.extend(cifar_classes[1])
        range_no_id = list(range(50000))
        for image in self.params['poison_images'] + self.params['poison_images_test']:
            if image in range_no_id:
                range_no_id.remove(image)

        # add random images to other parts of the batch
        for batches in range(0, self.params['size_of_secret_dataset']):
            range_iter = random.sample(range_no_id,
                                       self.params['batch_size'])
            # range_iter[0] = self.params['poison_images'][0]
            indices.extend(range_iter)
            # range_iter = random.sample(range_no_id,
            #            self.params['batch_size']
            #                -len(self.params['poison_images'])*self.params['poisoning_per_batch'])
            # for i in range(0, self.params['poisoning_per_batch']):
            #     indices.extend(self.params['poison_images'])
            # indices.extend(range_iter)
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def poison_test_dataset(self):
        #
        # return [(self.train_dataset[self.params['poison_image_id']][0],
        # torch.IntTensor(self.params['poison_label_swap']))]
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range(1000)
                           ))


    def load_data(self,noniid=1,Dclient=20):
        logger.info('Loading data')


        ### data load
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if noniid==1:
            # feature shift
            train_loaders, test_loaders = generate_Digit5('data/Digit5/')
        elif noniid==2:
            # label shift
            train_loaders, test_loaders = generate_shift_Digit5('data/Digit5/', Dclient=Dclient)
        elif noniid==3:
            # feature and label shift
            train_loaders,test_loaders=generate_shift_Digit5('data/Digit5/',Dclient=Dclient)
        # train_loaders,test_loaders=generate_Digit5('data/Digit5/')
        # if self.params['sampling_dirichlet']:
        #     ## sample indices for participants using Dirichlet distribution
        #     indices_per_participant = self.sample_dirichlet_train_data(
        #         self.params['number_of_total_participants'],
        #         alpha=self.params['dirichlet_alpha'])
        #     train_loaders = [(pos, self.get_train(indices)) for pos, indices in
        #                      indices_per_participant.items()]
        # else:
        #     ## sample indices for participants that are equally
        #     # splitted to 500 images per participant
        #     all_range = np.argsort(self.train_dataset.targets.numpy()).tolist()
        #     # random.shuffle(all_range)
        #     mask=[self.train_dataset.targets.numpy().tolist().count(i) for i in range(47)]
        #     train_loaders = [(pos, self.get_train_old(all_range, pos,self.params['divide_q'],mask))
        #                      for pos in range(self.params['number_of_total_participants'])]



        self.train_data = train_loaders
        self.test_data = test_loaders
        # self.test_data = torch.utils.data.DataLoader(self.test_dataset,
        #                                           batch_size=self.params['test_batch_size'],
        #                                           shuffle=True)



        #####
        # self.poisoned_data_for_train = self.poison_dataset()
        # self.test_data_poison = self.poison_test_dataset()



        # self.params['adversary_list'] = [POISONED_PARTICIPANT_POS] + \
        #                            random.sample(range(len(train_loaders)),
        #                                          self.params['number_of_adversaries'] - 1)
        # logger.info(f"Poisoned following participants: {self.params['adversary_list']}")


    def get_train(self, indices):
        """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices))
        return train_loader

    def get_train_old(self, all_range, model_no,sq,mask):
        """
        This method equally splits the dataset.
        :param params:
        :param all_range:
        :param model_no:
        :return:
        """

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])

        # sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        area=int(model_no/10)
        sub_indices=[]
        p=int(sq*mask[area]/10)
        q=int((1-sq)*mask[area]/460)
        mask_index=[]
        mask_index.extend([0])
        for i in range(47):
            mask_index.extend([mask_index[-1]+mask[i]])
        for i in range(47):
            if area==i:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i+1]], size=p, replace=False))
            else:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i+1]], size=q, replace=False))


        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader


    def get_secret_loader(self):
        """
        For poisoning we can use a larger data set. I don't sample randomly, though.

        """
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        shuffled_indices = indices[:self.params['size_of_secret_dataset']]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=SubsetSampler(shuffled_indices))
        return train_loader

    def get_test(self,sq,area):
        all_range = np.argsort(self.test_dataset.targets.numpy()).tolist()
        # random.shuffle(all_range)
        mask = [self.test_dataset.targets.numpy().tolist().count(i) for i in range(47)]
        sub_indices = []
        p = int(sq * mask[area])
        q = int((mask[area] - p) / 46)
        mask_index = []
        mask_index.extend([0])
        for i in range(47):
            mask_index.extend([mask_index[-1] + mask[i]])
        for i in range(47):
            if area == i:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i + 1]], size=p, replace=False))
                # sub_indices.extend(all_range[mask_index[i]:mask_index[i]+p])
            else:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i + 1]], size=q, replace=False))
                # sub_indices.extend(all_range[mask_index[i+1]-(area*q)-q:mask_index[i+1]-(area*q)])


        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      sub_indices),
                                                  )
        return test_loader

    def get0_test(self):
        all_range = np.argsort(self.test_dataset.targets.numpy()).tolist()
        # random.shuffle(all_range)
        mask = [self.test_dataset.targets.numpy().tolist().count(i) for i in range(47)]
        sub_indices = []
        sq=0.5
        area=0
        p = int(sq * mask[area])
        q = int((mask[area] - p) / 47) + 1
        mask_index = []
        mask_index.extend([0])
        for i in range(47):
            mask_index.extend([mask_index[-1] + mask[i]])
        for i in range(47):
            if area==i:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i+1]], size=p, replace=False))
            else:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i+1]], size=q, replace=False))

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      sub_indices),
                                                  )

        return test_loader
    # def get0_test_only(self):
    #     all_range = np.argsort(self.test_dataset.targets.numpy()).tolist()
    #     mask = [self.test_dataset.targets.numpy().tolist().count(i) for i in range(47)]
    #     test_loader = torch.utils.data.DataLoader(self.test_dataset,
    #                                               batch_size=self.params['test_batch_size'],
    #                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
    #                                                   all_range[:mask[0]]),
    #                                               )
    #
    #     return test_loader
    # def get7_test_only(self):
    #     all_range = np.argsort(self.test_dataset.targets.numpy()).tolist()
    #     mask = [self.test_dataset.targets.numpy().tolist().count(i) for i in range(47)]
    #     mask_index = []
    #     mask_index.extend([0])
    #     for i in range(47):
    #         mask_index.extend([mask_index[-1] + mask[i]])
    #     test_loader = torch.utils.data.DataLoader(self.test_dataset,
    #                                               batch_size=self.params['test_batch_size'],
    #                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
    #                                                   all_range[mask_index[7]:mask_index[8]]),
    #                                               )
    #
    #     return test_loader
    def get7_test(self):
        all_range = np.argsort(self.test_dataset.targets.numpy()).tolist()
        # random.shuffle(all_range)
        mask = [self.test_dataset.targets.numpy().tolist().count(i) for i in range(47)]
        sub_indices = []
        sq = 0.5
        area = 7
        p = int(sq * mask[area])
        q = int((mask[area] - p) / 47) + 1
        mask_index = []
        mask_index.extend([0])
        for i in range(47):
            mask_index.extend([mask_index[-1] + mask[i]])
        for i in range(47):
            if area == i:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i + 1]], size=p, replace=False))
            else:
                sub_indices.extend(np.random.choice(all_range[mask_index[i]:mask_index[i + 1]], size=q, replace=False))

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      sub_indices),
                                                  )
        return test_loader
    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

