from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
os.environ['DISPLAY'] = ':0'
import torch.utils.data as data

class CIFAR100(data.Dataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        This is a subclass of the `CIFAR10` Dataset.
        """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,area=-1,only=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        if self.train:
            for i in range(20):
                with open('./data/cifar100/train_' + str(i), 'rb')as f:
                    entry2 = pickle.load(f, encoding='latin1')
                    self.data.append(entry2['data'])
                    self.targets.extend(entry2['fine_labels'])
        else:
            if only:
                with open('./data/cifar100/test_' + str(area), 'rb')as f:
                    entry2 = pickle.load(f, encoding='latin1')
                    self.data.append((entry2['data'])[:500])
                    self.targets.extend(entry2['fine_labels'][:500])
            else:
                if area >= 0:
                    for i in range(20):
                        with open('./data/cifar100/test_' + str(i), 'rb')as f:
                            entry2 = pickle.load(f, encoding='latin1')
                            if i == area:
                                self.data.append((entry2['data'])[:495])
                                self.targets.extend(entry2['fine_labels'][:495])
                            else:
                                self.data.append((entry2['data'])[23*i:23*i+45])
                                self.targets.extend(entry2['fine_labels'][23*i:23*i+45])
                else:
                    for i in range(20):
                        with open('./data/cifar100/test_' + str(i), 'rb')as f:
                            entry2 = pickle.load(f, encoding='latin1')
                            self.data.append(entry2['data'])
                            self.targets.extend(entry2['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()


    def _load_meta(self):
        path = os.path.join(self.root, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

