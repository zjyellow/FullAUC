# Intro
The offitial implement of "Full AUC Optimization for Open Set Recognition in Remote Sensing Images"

# env
CUDA==11.7  
GPU Memory >= 4GB
```
torch==1.13.0+cu117
torchaudio==0.13.0+cu117
torchvision==0.14.0+cu117
numpy==1.23.0
pandas==1.3.5
scipy==1.9.0
scikit-learn==1.0.2
scikit-image==0.24.0
```
ps: We suggest that the Numpy version should < 2 in this code.

# dataset
here provided five remote sensing datasets:[google drive](https://drive.google.com/file/d/1eWosXC8ktq0lfLGHemcG1YkXlcvwCM8o/view?usp=sharing)

# method
FullAUC & FullAUC-NF: `./loss/FullAUC.py`  
FullAUC-F : `./loss/FullAUC2.py`  
comparison loss functions also list in `loss` folder.

# usage
ref `fauc_train.sh`

# cite
please cite us if you used our code, dataset or our proposed method in your work.
```
@ARTICLE{11343875,
  author={Zhang, Xiao-Lei and Huang, Zijun and Zhang, Jichao and Cao, Zhe and Zhao, Lei and Xu, Menglong and Liu, Cheng-Lin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FullAUC Optimization for Open-Set Recognition in Remote Sensing Images}, 
  year={2026},
  volume={64},
  number={},
  pages={1-13},
  keywords={Remote sensing;Prototypes;Optimization;Image classification;Training;Classification algorithms;Image recognition;Accuracy;Reviews;Deep learning;AUC optimization;background class regularization;open-set recognition (OSR);remote sensing image classification},
  doi={10.1109/TGRS.2026.3652236}}
```


# acknowledgement
Our codes are based on the repositories [Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI'21)](https://github.com/iCGY96/ARPL) and [OpenAUC: Towards AUC-Oriented Open-Set Recognition](https://github.com/wang22ti/OpenAUC)



