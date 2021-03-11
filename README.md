# Stagewise Training Accelerates Convergence of Testing Error Over SGD  [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://proceedings.neurips.cc/paper/2019/file/fcdf25d6e191893e705819b177cddea0-Paper.pdf)

This is the offical implementation of paper "**Stagewise Training Accelerates Convergence of Testing Error Over SGD**" published on **NeurIPS 2019**. 

## Installation
```
Python=3.5
Numpy=1.18.5 
Scipy=1.2.1
Scikit-Learn=0.20.3
Pillow=5.0.0
Tensorflow>=1.10.0
```

### Run
#### To run SGD with c/t, c/sqrt(t), stagewiseSGD_v1, you can specify version = 0, 1, 2
```
python main_SGD.py --version=2 --lr=0.7 --is_save_model=False --use_L2=False --activation='elu' --model='resnet' --resnet_layers=56 --dataset=100 --random_seed=789 --num_iters=80040
```
#### To run SGD with stagewiseSGD_v2, stagewiseSGD_v3, you can specify version = 3, 4
```
python main_START.py --version=3 --lr=0.3 --gamma=5000 --use_L2=False --is_save_model=False --activation='elu' --model='resnet' --resnet_layers=56 --dataset=10 --random_seed=123 --num_iters=80040
```
#### To compute theta/mu: (Prior to this step, you need to train and save the models by setting --is_save_model=False=True --num_iters=200400 iterations in main_*.py)
```
python eval_compute_theta_mu.py --version=2 --use_L2=False --model='resnet' --resnet_layers=56 --dataset=10 --random_seed=123
```

#### To compute minimal eigen value (Lanczos Method): (If you don't specify --model_iter, the program will evaluate [20000, 40000, 60000, 80000])
```
python eval_compute_min_eig_val.py --lr=0.001 --use_L2=False --activation='elu' --model='resnet' --resnet_layers=20 --random_seed=123 --model_iter=80000
```

### Hyperparameter tuning
```
gamma=[500, 1000, 2000, ...]
lr = [0.1, 0.01, 0.001, ...]
```

## Bibtex 
If you use this repository in your work, please cite our paper:

```
@inproceedings{NEURIPS2019_fcdf25d6,
 author = {Yuan, Zhuoning and Yan, Yan and Jin, Rong and Yang, Tianbao},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Stagewise Training Accelerates Convergence of Testing Error Over SGD},
 url = {https://proceedings.neurips.cc/paper/2019/file/fcdf25d6e191893e705819b177cddea0-Paper.pdf},
 volume = {32},
 year = {2019}
}
```
