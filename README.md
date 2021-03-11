# Stagewise SGD  [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://proceedings.neurips.cc/paper/2019/file/fcdf25d6e191893e705819b177cddea0-Paper.pdf)

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

### Usage

#### (1) SGD with c/t, c/sqrt(t)
You can run this command for SGD and set `--version=0,1` for choosing different learning rate schedulers.
```
python main_SGD.py --version=2 --lr=0.7 --is_save_model=False --use_L2=False --activation='elu' --model='resnet' --resnet_layers=56 --dataset=100 --random_seed=789 --num_iters=80040
```
#### (2) StagewiseSGD V1, StagewiseSGD V2, StagewiseSGD V3
You can run this command for Stagewise SGD and set `--version=2,3,4` for Stagewise SGD V1, V2, V3.
```
python main_START.py --version=3 --lr=0.3 --gamma=5000 --use_L2=False --is_save_model=False --activation='elu' --model='resnet' --resnet_layers=56 --dataset=10 --random_seed=123 --num_iters=80040
```
#### (3) Theta/Mu
You can run this command for computing theta/mu value. Note that prior to this step, you need to train and save the models by setting `--is_save_model=False=True --num_iters=200400` from previous steps. 
```
python eval_compute_theta_mu.py --version=2 --use_L2=False --model='resnet' --resnet_layers=56 --dataset=10 --random_seed=123
```

#### (4) Minimal Eigen Value (Lanczos Method): 
You can run this command for computing minimal eigen value using Lanczos Method. Note that if you don't specify `--model_iter`, the code will evaluate your models at iteartion 20000, 40000, 60000, 80000.
```
python eval_compute_min_eig_val.py --lr=0.001 --use_L2=False --activation='elu' --model='resnet' --resnet_layers=20 --random_seed=123 --model_iter=80000
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
