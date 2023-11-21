# CFLCP

Implementation of the paper : Clustered Federated Learning Based on Client's Prototypes. (under review)

## Requirments
The experiments were run on NVIDIA Tesla P100 using cuda10.2.
This code requires the following:
* Python 3.6
* PyTorch 1.10.0+cu102
* Torchvision 0.11.0+cu102
* Numpy 1.18.5
* Scikit-learn 0.24.2
* scipy 1.4.1
* PyYAML 5.4.1
* visdom 0.1.8.9

## Data Preparation
* Experiments are run on Digits-five and Office-Caltech-10.

## Running the experiments
All experiments are shown in auto.sh.

For example:
* To train the CFLCP on Digits-five with 100 clients under feature shift Non-IID setting:
```
python training_cflcp.py --params=utils/digits5_params.yaml --sigma=0.8 --noniid=1
```
* To train the CFLCP on Digits-five(MNIST-M) with 20 clients under label shift Non-IID setting(MNIST-M):
```
python training_cflcp.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2
```
* To train the CFLCP on Digits-five(MNIST) with 20 clients under label shift Non-IID setting:
```
python training_cflcp.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2
```
* To train the CFLCP on Digits-five with 100 clients under feature and label shift Non-IID setting:
```
python training_cflcp.py --params=utils/digits5_params.yaml --sigma=0.8 --noniid=3
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
Details are given some of those parameters:

* ```--params:```  Full parameter file. Options: 'utils/digits5_params.yaml', 'utils/office_params.yaml'
* ```--sigma:```  Aggregate weights between clusters. Default: 0.8
* ```--noniid:```     Non-IID setting: 1(Feature Shift), 2(Label Shift), 3(Feature and Label Shift). Default: 1. Options:1,2,3
* ```--domain:```     The dataset used in label shift setting. Default: 0. Options: 0,1,2,3,4  (MNIST-M, MNIST, SYN, USPS, SVHN)

## Citation
If you find this project helpful, please consider to cite the following paper:
```
under review
```
