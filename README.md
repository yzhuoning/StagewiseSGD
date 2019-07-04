# Stagewise_SGD


Requirements
--
Python3.5.4, Pandas, Numpy, Scipy, Tensorflow(1.10.0)

Usage
--
For first-time running, the program will automatically download CIFAR10, CIFAR100...

Linux/MacOS:

To run SGD with c/t, c/sqrt(t), stagewiseSGD_v1, you can specify version = 0, 1, 2
--
python main_SGD.py --version=2 --lr=0.7 --is_save_model=False --use_L2=False --activation='elu' --model='resnet' --resnet_layers=56 --dataset=100 --random_seed=789 --num_iters=80040


To run SGD with stagewiseSGD_v2, stagewiseSGD_v3, you can specify version = 3, 4
--
python main_START.py --version=3 --lr=0.3 --gamma=5000 --use_L2=False --is_save_model=False --activation='elu' --model='resnet' --resnet_layers=56 --dataset=10 --random_seed=123 --num_iters=80040


To compute theta/mu:
(Prior to this step, you need to train and save the models by setting --is_save_model=False=True --num_iters=200400 iterations in main_*.py)
--
python eval_compute_theta_mu.py --version=2 --use_L2=False --model='resnet' --resnet_layers=56 --dataset=10 --random_seed=123


To compute minimal eigen value (Lanczos Method):
(If you don't specify --model_iter, the program will evaluate [20000, 40000, 60000, 80000])
--
python eval_compute_min_eig_val.py --lr=0.001 --use_L2=False --activation='elu' --model='resnet' --resnet_layers=20 --random_seed=123 --model_iter=80000 


Windows:
Anaconda Python3 (recommended)


Arguments 
--

main_SGD.py, main_START.py:
--
    tf.app.flags.DEFINE_string('model', 'resnet', '''which model to train: resnet or convnet''')
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''random seed for initializations''')
    tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
    tf.app.flags.DEFINE_float('lr', 0.1, '''learning rate to train the models''')
    tf.app.flags.DEFINE_integer('dataset', 10, '''dataset to evalute: 10 or 100''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_integer('gamma', 5000, '''gamma for START, leave this any values if train other models''')
    tf.app.flags.DEFINE_integer('num_iters', 230*400, '''total number of iterations to train the model''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data''')
    tf.app.flags.DEFINE_boolean('is_save_model', False, '''whether to save model or not ''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer''')
    tf.app.flags.DEFINE_integer('version', 3, '''[3:SGDv2, 4:SGDv3] (don't input other numbers)''')
	
	
eval_compute_min_eig_val.py:
--
    tf.app.flags.DEFINE_string('model', 'resnet', '''which model to train: resnet or convnet''')
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''input the same random seed used for training models''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data. *only use in training models''') 
    tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
    tf.app.flags.DEFINE_float('lr', 1e-3, '''learning rate to compute the min-eig-val, differnt from previous learning rate''')
    tf.app.flags.DEFINE_integer('dataset', 10, '''dataset to evalute: 10 or 100''')
    tf.app.flags.DEFINE_integer('model_iter', 80000, '''e.g. 80000 means to compute min-eig-val using model at 80000-iteration and default: 99 (evalute all saved models) ''')
    tf.app.flags.DEFINE_integer('resnet_layers', 3, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_boolean('use_L2', True, '''whether to use L2 regularizer to compute gradient and loss''')
    tf.app.flags.DEFINE_integer('version', 2, '''[0: c/t, 1: c/sqrt(t),  2:SGDv1](don't input other numbers)''')

	
eval_compute_theta_mu.py:
--
    tf.app.flags.DEFINE_string('model', 'resnet', '''model to train: resnet or convnet''')
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''input the same random seed used for training models''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data. *only use in training models''')
    tf.app.flags.DEFINE_float('lr', 0.1, '''doing nothing here''')
    tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
    tf.app.flags.DEFINE_integer('dataset', 100, '''dataset to evalute''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer to compute gradient and loss''')
    tf.app.flags.DEFINE_integer('version', 2, '''2:SGDv1 (we test stagewiseV1 only)''')
    
