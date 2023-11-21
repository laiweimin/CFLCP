import copy
from collections import defaultdict
from  generate_Digit5 import generate_Digit5
from  generate_Office10 import generate_Office10, generate_shift_Office10
import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np
from utils import cifar
from models.resnet import ResNet18
from models.simple import SimpleMnist,Digit5CNN,Office10CNN
from models.word_model import RNNModel
from utils.text_load import *
from utils.utils import SubsetSampler
from models.MOON_model import *
from torch import nn
logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 2

device="cuda"

class ImageHelper(Helper):


    def poison(self):
        return

    def create_model(self,single=False,device="cuda",base_model='resnet18',out_dim=256):
        # top_model = ModelFedCon(base_model,out_dim,n_classes=10,name='Top',created_time=self.params['current_time'])
        # local_model = ModelFedCon(base_model,out_dim,n_classes=10,name='Local',created_time=self.params['current_time'])
        # target_model = ModelFedCon(base_model,out_dim,n_classes=10,name='Target',created_time=self.params['current_time'])
        logger.info('Creating model')
        top_model = Office10CNN(num_input=3, num_classes=10, name='Top', created_time=self.params['current_time'])
        local_model = Office10CNN(num_input=3, num_classes=10, name='Local', created_time=self.params['current_time'])
        target_model = Office10CNN(num_input=3, num_classes=10, name='Target', created_time=self.params['current_time'])

        local_model.to(device)
        target_model.to(device)
        top_model.to(device)
        logger.info('Creating model Done')
        model_list=[]
        for i in range(10):
            # model_p = ModelFedCon(base_model, out_dim, n_classes=10, name='Target'+str(i),
            #                             created_time=self.params['current_time'])
            model_p = Office10CNN(num_input=3,num_classes=10, name='Target'+str(i),
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


    def load_data(self,noniid=1):
        logger.info('Loading data')


        if noniid==1:
            # feature shift
            train_loaders,test_loaders=generate_Office10("data/office_caltech_10/")
        else:
            # feature and label shift
            train_loaders,test_loaders=generate_shift_Office10("data/office_caltech_10/")

        self.train_data = train_loaders
        self.test_data = test_loaders

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.to(device)
        target = target.to(device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

