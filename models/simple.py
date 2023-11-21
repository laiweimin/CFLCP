import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import datetime


class SimpleNet(nn.Module):
    def __init__(self, name='SimpleNet', created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name



    def visualize(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None,dataset_name=''):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        name=name+dataset_name
        if isinstance(acc,torch.Tensor):
            vis.line(X=np.array([epoch]), Y=torch.stack([acc]), name=name, win='vacc_{0}'.format(self.created_time),
                     env=eid,
                     update='append' if vis.win_exists('vacc_{0}'.format(self.created_time), env=eid) else None,
                     opts=dict(showlegend=True, title='Accuracy_{0}'.format(self.created_time),
                               width=700, height=400))
        else:
            vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='vacc_{0}'.format(self.created_time),
                     env=eid,
                     update='append' if vis.win_exists('vacc_{0}'.format(self.created_time), env=eid) else None,
                     opts=dict(showlegend=True, title='Accuracy_{0}'.format(self.created_time),
                               width=700, height=400))

        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                                     win='vloss_{0}'.format(self.created_time),
                                     update='append' if vis.win_exists('vloss_{0}'.format(self.created_time), env=eid) else None,
                                     opts=dict(showlegend=True, title='Loss_{0}'.format(self.created_time), width=700, height=400))

        return



    def train_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='vtrain'):
        if isinstance(loss, torch.Tensor):
            vis.line(X=np.array([(epoch - 1) * data_len + batch]), Y=torch.stack([loss]),
                     env=eid,
                     name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                     update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                     opts=dict(showlegend=True, width=700, height=400,
                               title='Train loss_{0}'.format(self.created_time)))
        else:
            vis.line(X=np.array([(epoch - 1) * data_len + batch]), Y=np.array([loss]),
                     env=eid,
                     name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                     update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                     opts=dict(showlegend=True, width=700, height=400,
                               title='Train loss_{0}'.format(self.created_time)))




    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)


    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                # shape = param.shape
                #
                # random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(
                #     torch.cuda.FloatTensor)
                # negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())




class SimpleMnist(SimpleNet):
    def __init__(self, num_input=1,num_classes=47,name=None, created_time=None):
        super(SimpleMnist, self).__init__(name, created_time)
        self.conv1 = nn.Conv2d(num_input, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, num_classes)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Digit5CNN(SimpleNet):
    def __init__(self, num_input=3,num_classes=10,name=None, created_time=None):
        super(Digit5CNN, self).__init__(name, created_time)
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn1", nn.BatchNorm2d(16))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv2", nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn2", nn.BatchNorm2d(32))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv3", nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn3", nn.BatchNorm2d(64))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(4096, 2048))
        self.linear.add_module("bn4", nn.BatchNorm1d(2048))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(2048, 512))
        self.linear.add_module("bn5", nn.BatchNorm1d(512))
        self.linear.add_module("relu5", nn.ReLU())
        self.linear.add_module("dropout2", nn.Dropout())
        self.linear.add_module("fc3", nn.Linear(512, 256))
        self.linear.add_module("bn6", nn.BatchNorm1d(256))
        self.linear.add_module("relu6", nn.ReLU())

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return feature,feature,F.log_softmax(out, dim=1)

class Office10CNN(SimpleNet):
    def __init__(self, num_input=3,num_classes=10,name=None, created_time=None):
        super(Office10CNN, self).__init__(name, created_time)
        self.encoder = nn.Sequential()
        self.encoder.add_module("conv1", nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn1", nn.BatchNorm2d(64))
        self.encoder.add_module("relu1", nn.ReLU())
        self.encoder.add_module("maxpool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv2", nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn2", nn.BatchNorm2d(64))
        self.encoder.add_module("relu2", nn.ReLU())
        self.encoder.add_module("maxpool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
        self.encoder.add_module("conv3", nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2))
        self.encoder.add_module("bn3", nn.BatchNorm2d(128))
        self.encoder.add_module("relu3", nn.ReLU())

        self.linear = nn.Sequential()
        self.linear.add_module("fc1", nn.Linear(8192, 3072))
        self.linear.add_module("bn4", nn.BatchNorm1d(3072))
        self.linear.add_module("relu4", nn.ReLU())
        self.linear.add_module("dropout", nn.Dropout())
        self.linear.add_module("fc2", nn.Linear(3072, 2048))
        self.linear.add_module("bn5", nn.BatchNorm1d(2048))
        self.linear.add_module("relu5", nn.ReLU())
        self.linear.add_module("dropout2", nn.Dropout())
        self.linear.add_module("fc3", nn.Linear(2048, 512))
        self.linear.add_module("bn6", nn.BatchNorm1d(512))
        self.linear.add_module("relu6", nn.ReLU())


        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        feature = self.encoder(x)
        feature = feature.view(batch_size, -1)
        feature = self.linear(feature)
        out = self.fc(feature)
        return feature,feature,F.log_softmax(out, dim=1)