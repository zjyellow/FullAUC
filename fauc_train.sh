#!/bin/bash

# nohup python osr5_baseline.py --dataset nwpu --gpu 0 --loss FAUCLoss --max-epoch 600 --lr 0.005 > OVRN_nwpu.log &
# nohup python osr5_baseline.py --dataset RSSCN7 --gpu 1 --loss FAUCLoss --max-epoch 600 --lr 0.005 > OVRN_RSSCN7.log &
# nohup python osr5_baseline.py --dataset siri --gpu 2 --loss FAUCLoss --max-epoch 600 --lr 0.005 > OVRN_siri.log &
# nohup python osr5_baseline.py --dataset AID --gpu 3 --loss FAUCLoss --max-epoch 600 --lr 0.005 > OVRN_AID.log &
# nohup python osr5_baseline.py --dataset EuroSAT --gpu 0 --loss FAUCLoss --max-epoch 600 --lr 0.005 > OVRN_EuroSAT.log &
# nohup python ood_baseline.py --gpu 1 --dataset nwpu_ood --out-dataset EuroSAT_ood --gpu 2 --loss FAUCLoss  --lr 0.005 > OVRN_ood_nwpuood_EuroSATood.log &
# nohup python ood_baseline.py --gpu 2 --dataset RSSCN7_ood --out-dataset siri_ood --gpu 3 --loss FAUCLoss --lr 0.005 > OVRN_ood_RSSCN7ood_siriood.log &

# nohup python generalized_osr_baseline.py --dataset nwpu --gpu 0 --loss FAUCLoss2 --lr 0.0001 --max-epoch 600 > fauc_logs/fauc-f_nwpu8.log &
# nohup python generalized_osr_baseline.py --dataset siri --gpu 3 --loss FAUCLoss2 --lr 0.0001 --max-epoch 600 > fauc_logs/fauc-f_siri8.log &
# nohup python generalized_osr_baseline.py --dataset AID --gpu 1 --loss FAUCLoss2 --lr 0.0001 --max-epoch 600 > fauc_logs/fauc-f_AID8.log &
# nohup python generalized_osr_baseline.py --dataset EuroSAT --gpu 2 --loss FAUCLoss2 --lr 0.0001 --max-epoch 600 > fauc_logs/fauc-f_EuroSAT8.log &
# nohup python generalized_osr_baseline.py --dataset RSSCN7 --gpu 3 --loss FAUCLoss2 --lr 0.0001 --max-epoch 600 > fauc_logs/fauc-f_RSSCN78.log &
