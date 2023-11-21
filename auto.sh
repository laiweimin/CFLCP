#!/bin/bash

nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=1 > log/cflcp_noniid_1_0.8.txt 2>&1 &
nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2 > log/cflcp_noniid_2_MM_0.8.txt 2>&1 &
nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2 > log/cflcp_noniid_2_Mn_0.8.txt 2>&1 &
nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=2 --sigma=0.8 --noniid=2 > log/cflcp_noniid_2_sy_0.8.txt 2>&1 &
nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=3 --sigma=0.8 --noniid=2 > log/cflcp_noniid_2_us_0.8.txt 2>&1 &
nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=4 --sigma=0.8 --noniid=2 > log/cflcp_noniid_2_sv_0.8.txt 2>&1 &
nohup python -u training_cflcp.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=3 > log/cflcp_noniid_3_0.8.txt 2>&1 &

nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=1 > log/fedavg_noniid_1.txt 2>&1 &
nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2 > log/fedavg_noniid_2_MM.txt 2>&1 &
nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2 > log/fedavg_noniid_2_Mn.txt 2>&1 &
nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=2 --sigma=0.8 --noniid=2 > log/fedavg_noniid_2_sy.txt 2>&1 &
nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=3 --sigma=0.8 --noniid=2 > log/fedavg_noniid_2_us.txt 2>&1 &
nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=4 --sigma=0.8 --noniid=2 > log/fedavg_noniid_2_sv.txt 2>&1 &
nohup python -u training_fedavg.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=3 > log/fedavg_noniid_3.txt 2>&1 &

nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=1 > log/fedprox_noniid_1.txt 2>&1 &
nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2 > log/fedprox_noniid_2_MM.txt 2>&1 &
nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2 > log/fedprox_noniid_2_Mn.txt 2>&1 &
nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=2 --sigma=0.8 --noniid=2 > log/fedprox_noniid_2_sy.txt 2>&1 &
nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=3 --sigma=0.8 --noniid=2 > log/fedprox_noniid_2_us.txt 2>&1 &
nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=4 --sigma=0.8 --noniid=2 > log/fedprox_noniid_2_sv.txt 2>&1 &
nohup python -u training_fedprox.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=3 > log/fedprox_noniid_3.txt 2>&1 &

nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=1 > log/ifca_noniid_1.txt 2>&1 &
nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2 > log/ifca_noniid_2_MM.txt 2>&1 &
nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2 > log/ifca_noniid_2_Mn.txt 2>&1 &
nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=2 --sigma=0.8 --noniid=2 > log/ifca_noniid_2_sy.txt 2>&1 &
nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=3 --sigma=0.8 --noniid=2 > log/ifca_noniid_2_us.txt 2>&1 &
nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=4 --sigma=0.8 --noniid=2 > log/ifca_noniid_2_sv.txt 2>&1 &
nohup python -u training_ifca.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=3 > log/ifca_noniid_3.txt 2>&1 &

nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=1 > log/flexcfl_noniid_1.txt 2>&1 &
nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2 > log/flexcfl_noniid_2_MM.txt 2>&1 &
nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2 > log/flexcfl_noniid_2_Mn.txt 2>&1 &
nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=2 --sigma=0.8 --noniid=2 > log/flexcfl_noniid_2_sy.txt 2>&1 &
nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=3 --sigma=0.8 --noniid=2 > log/flexcfl_noniid_2_us.txt 2>&1 &
nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=4 --sigma=0.8 --noniid=2 > log/flexcfl_noniid_2_sv.txt 2>&1 &
nohup python -u training_FlexCFL.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=3 > log/flexcfl_noniid_3.txt 2>&1 &

nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=3 > log/fedsoft_noniid_3.txt 2>&1 &
nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=2 > log/fedsoft_noniid_2_MM.txt 2>&1 &
nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=1 --sigma=0.8 --noniid=2 > log/fedsoft_noniid_2_Mn.txt 2>&1 &
nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=2 --sigma=0.8 --noniid=2 > log/fedsoft_noniid_2_sy.txt 2>&1 &
nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=3 --sigma=0.8 --noniid=2 > log/fedsoft_noniid_2_us.txt 2>&1 &
nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=4 --sigma=0.8 --noniid=2 > log/fedsoft_noniid_2_sv.txt 2>&1 &
nohup python -u training_fedsoft.py --params=utils/digits5_params.yaml --domain=0 --sigma=0.8 --noniid=1 > log/fedsoft_noniid_1.txt 2>&1 &




nohup python -u training_office10_cflcp.py --params=utils/office_params.yaml --domain=0 --noniid=3 > log/cflcp_office10_noniid_3.txt 2>&1 &
nohup python -u training_office10_cflcp.py --params=utils/office_params.yaml --domain=0 --noniid=1 > log/cflcp_office10_noniid_1.txt 2>&1 &
nohup python -u training_office10_fedavg.py --params=utils/office_params.yaml --domain=0 --noniid=3 > log/fedavg_office10_noniid_3.txt 2>&1 &
nohup python -u training_office10_fedavg.py --params=utils/office_params.yaml --domain=0 --noniid=1 > log/fedavg_office10_noniid_1.txt 2>&1 &
nohup python -u training_office10_fedprox.py --params=utils/office_params.yaml --domain=0 --noniid=3 > log/fedprox_office10_noniid_3.txt 2>&1 &
nohup python -u training_office10_fedprox.py --params=utils/office_params.yaml --domain=0 --noniid=1 > log/fedprox_office10_noniid_1.txt 2>&1 &
nohup python -u training_office10_ifca.py --params=utils/office_params.yaml --domain=0 --noniid=3 > log/ifca_office10_noniid_3.txt 2>&1 &
nohup python -u training_office10_ifca.py --params=utils/office_params.yaml --domain=0 --noniid=1 > log/ifca_office10_noniid_1.txt 2>&1 &
nohup python -u training_office10_FlexCFL.py --params=utils/office_params.yaml --domain=0 --noniid=3 > log/flexcfl_office10_noniid_3.txt 2>&1 &
nohup python -u training_office10_FlexCFL.py --params=utils/office_params.yaml --domain=0 --noniid=1 > log/flexcfl_office10_noniid_1.txt 2>&1 &