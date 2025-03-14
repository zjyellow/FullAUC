# intro
The offitial implement of "Full AUC Optimization for Open Set Recognition in Remote Sensing Images"

# env
CUDA==11.7
GPU Memory >= 6GB

torch==1.13.0+cu117
torchaudio==0.13.0+cu117
torchvision==0.14.0+cu117
numpy==1.23.0
pandas==1.3.5
scipy==1.9.0
scikit-learn==1.0.2
scikit-image==0.24.0

ps: We suggest that the Numpy version should < 2 in this code.

# method
FullAUC & FullAUC-NF: ./loss/FullAUC.py
FullAUC-F : ./loss/FullAUC2.py

# usage
ref fauc_train.sh

# acknowledgement
Our codes are based on the repositories [Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI'21)](https://github.com/iCGY96/ARPL) and [OpenAUC: Towards AUC-Oriented Open-Set Recognition](https://github.com/wang22ti/OpenAUC)




