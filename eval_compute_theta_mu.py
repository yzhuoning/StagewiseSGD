import tensorflow as tf
import numpy as np
import os 
from datetime import datetime
from numpy import linalg as LA
from convnet import convnet_inference
from resnet_model import resnet_inference
from os import listdir
import pandas as pd
import cifar_input as cifar_data
import my_utils

tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.reset_default_graph()

try:
    FLAGS.activation
except:
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('model', 'resnet', '''which model to train: resnet or convnet''')
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''input the same random seed used for training models''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data. *only use in training models''') # don't change this here
    tf.app.flags.DEFINE_float('lr', 0.1, '''doing nothing here''') # don't change this here
    tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
    tf.app.flags.DEFINE_integer('dataset', 100, '''dataset to evalute''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer  ''')
    tf.app.flags.DEFINE_integer('version', 2, '''[0: c/t, 1: c/sqrt(t),  2:SGDv1]''')
    

print ('-'*20 + '\nEvaluations on MU & Theta...\n' + '-'*20)

# Training Parameters
initial_learning_rate = FLAGS.lr
batch_size = FLAGS.train_batch_size
inference = resnet_inference if FLAGS.model == 'resnet'else convnet_inference

# tf Graph input
X = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
Y = tf.placeholder(tf.float32, [batch_size,])
phase_train = tf.placeholder(tf.bool, name='phase_train')

# do inference
logits = inference(X, num_classes=FLAGS.dataset, num_layers=FLAGS.resnet_layers, activations=FLAGS.activation, phase_train=phase_train) # when resnet you need to pass number of layers 

# Define loss and optimizer
W = [var for var in tf.trainable_variables ()]
loss_op = my_utils.cross_entropy_loss_with_l2(logits, Y, W, use_L2=FLAGS.use_L2)

# Call: gradients 
grads = tf.gradients(loss_op, W)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=5000)

# Build an initialization operation to run below.
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init)
    

# we don't need to load data every in the current session
cifar_data_dir = './cifar%d_data/raw_data_C%d.npy'%(FLAGS.dataset, FLAGS.dataset)
cifar_label_dir = './cifar%d_data/raw_label_C%d.npy'%(FLAGS.dataset, FLAGS.dataset)
if os.path.isfile(cifar_data_dir) and os.path.isfile(cifar_label_dir):
    raw_data = np.load(cifar_data_dir)
    raw_label = np.load(cifar_label_dir)
else:
    (raw_data, raw_label), (test_data, test_labels) = cifar_data.load_data(FLAGS.dataset, FLAGS.is_tune)
    np.save('./cifar%d_data/raw_data_C%d.npy'%(FLAGS.dataset, FLAGS.dataset), raw_data)
    np.save('./cifar%d_data/raw_label_C%d.npy'%(FLAGS.dataset, FLAGS.dataset), raw_label)

print ('load dataset: [CIFAR%d]'%FLAGS.dataset)  
num_batches = raw_data.shape[0]//batch_size

# read all models
random_seed = FLAGS.random_seed
checkpoint_dir = '../ImprovedICLR_v2/stagewise_sgd/models_%s-%d_v%d_%s_L2_%s/C%d/exp_%d/'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2), FLAGS.dataset, FLAGS.random_seed)
checkpoint_dir = './models_%s-%d_v%d_%s_L2_%s/C%d/exp_%d/'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2), FLAGS.dataset, FLAGS.random_seed)

model_dir = [checkpoint_dir + f.split('.data')[0] for f in listdir(checkpoint_dir) if 'data-' in f ] #and '120000' in f]
model_dir = my_utils.natural_sort(model_dir)
mode_optiomal_dir = model_dir[-1]
mode_optiomal_dir = [m for m in model_dir if '200000' in m ][0]
model_dir = model_dir[:-1]

# get W optimal 
saver.restore(sess, mode_optiomal_dir)
load_iter = int(mode_optiomal_dir.split('-')[-1])
W_opt = sess.run(W)
print ('W optimal: %.5f'%(W_opt[0].sum()))
loss_W_optimal = []
for n in range(num_batches):
    offset = (n) * batch_size
    print ('\rmodel-[%d]-batch-[%d]'%(load_iter, n), end='\r')
    train_batch_data = raw_data[offset:offset+batch_size, ...]
    train_batch_labels = raw_label[offset:offset+batch_size]
    feed_dict = {X: train_batch_data, Y:train_batch_labels, phase_train:False}
        
    loss_w_optimal_n = sess.run(loss_op, feed_dict)
    loss_W_optimal.append(loss_w_optimal_n)

loss_opt = np.mean(loss_W_optimal)  # 0.00001 #
print('model*-[%d]-optimal_loss: %.5f\n'%(load_iter, loss_opt))

save_csv = []
log_iter = []
log_ratio = []
log_mu = []
for idx, model__ in enumerate(model_dir):
    load_iter = int(model__.split('-')[-1])
    
    # uncomment the below lines if you want to check less number of points 
    #if load_iter not in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]:
    #    continue 
    
    saver.restore(sess, model__)
    W_t = sess.run(W)

    ratio, loss_W_current = [] ,[]
    mean_grad_sum = [np.zeros((w.shape.as_list())) for w in W]
    for n in range(num_batches):
      
        offset = (n) * batch_size
        print ('\rmodel-[%d]-batch-[%d]'%(load_iter, n), end='\r')
        train_batch_data = raw_data[offset:offset+batch_size, ...]
        train_batch_labels = raw_label[offset:offset+batch_size]
        feed_dict = {X: train_batch_data, Y:train_batch_labels, phase_train:False}

        grads_n  = sess.run(grads, feed_dict)
        loss_n = sess.run(loss_op, feed_dict)
        
        # compute mean gradients for each layer (divide by "N"(#batches) after for loop)
        mean_grad_sum = my_utils.cumulative_sum(mean_grad_sum, grads_n)

        # save resluts 
        loss_W_current.append(loss_n)
 
    loss_t = np.mean(loss_W_current)
    loss_l2_square_t = np.sum([np.square(LA.norm(g_mean/num_batches)) for g_mean in mean_grad_sum ])
    ratio_t = np.sum([np.inner(g.flatten()/num_batches, (w_t - w_opt).flatten()) for g, w_t, w_opt in zip(mean_grad_sum, W_t, W_opt) ]) 
        
    # compute theta 
    pl_i  = (loss_l2_square_t)/(loss_t - loss_opt)
    ratio_i = ratio_t/(loss_t - loss_opt)

    # compute mu
    w_diff_norm_square = np.sum([(np.square(LA.norm(w_t - w_opt))) for w_t, w_opt in zip(W_t, W_opt)])
    estimated_mu  = (loss_t - loss_opt)/(w_diff_norm_square*2)

    result_outptus = ('model-[%d]-PL:, %.5f, Ratio:, %.5f, grads_l2:%.5f, loss_t: %.5f, mu:, %.5f '%( load_iter, pl_i, ratio_i, loss_l2_square_t, loss_t, estimated_mu))
    print(result_outptus)
    log_iter.append(load_iter)
    log_ratio.append(ratio_i)
    log_mu.append(estimated_mu)
    df = pd.DataFrame(data={'model':log_iter, 'Ratio':log_ratio, 'mu':log_mu})
    if not os.path.exists('./logs_eval/'):
        os.makedirs('./logs_eval/')      
    df.to_csv('./logs_eval/%s-%d-v%d_%s_C%d_theta_mu_use_L2_%s_exp_%d.csv'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, FLAGS.dataset, str(FLAGS.use_L2), FLAGS.random_seed))
    
    
    
    