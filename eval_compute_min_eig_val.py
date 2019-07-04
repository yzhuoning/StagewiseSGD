from scipy import zeros, dot, random, mat, linalg, diag, sqrt, sum, hstack, ones
from scipy.linalg import norm, eig
from scipy.sparse.linalg import eigs
import numpy as np
import tensorflow as tf
from datetime import datetime
from os import listdir
import os
from convnet import convnet_inference
from resnet_model import resnet_inference
import pandas as pd
import cifar_input as cifar_data
import my_utils
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.reset_default_graph()

try:
    FLAGS.model_iter
except:
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('model', 'resnet', '''which model to train: resnet or convnet''')
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''input the same random seed used for training models''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data. *only use in training models''')
    tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
    tf.app.flags.DEFINE_float('lr', 1e-3, '''learning rate to compute the min-eig-val, which is differnt from lr for training model. * IF model doesn't converage, try smaller value ''')
    tf.app.flags.DEFINE_integer('dataset', 10, '''dataset to evalute''')
    tf.app.flags.DEFINE_integer('model_iter', 99, '''model at i-iteration to compute min-eig-val, e.g. default: 99 (evalute all saved models) ''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer  ''')
    tf.app.flags.DEFINE_integer('version', 2, '''[0: c/t, 1: c/sqrt(t),  2:SGDv1]''')
    

def compute_hv(grads1, grads2):
        hv = []
        for g1, g2 in zip(grads1, grads2):
            hv__ = (g2 - g1)/r
            hv.append(hv__)
        return hv
    
def Hv(u):
    update_direction_vars_ops = [w.assign(w + r*u_) for w, u_ in zip(W, u)]
    update_original_vars_ops = [w.assign(w - r*u_) for w, u_ in zip(W, u)]
    mean_hv_value = [np.zeros(u_.shape) for u_ in u]
    for n in range(num_batches):
        print ('\rmodel-[%d]-batch-[%d]'%(load_iter, n), end='\r')
        # deterministic version
        offset = (n) * batch_size
        train_batch_data = raw_data[offset:offset+batch_size, ...]
        if train_batch_data.shape[0] != batch_size:
            continue 
        train_batch_labels = raw_label[offset:offset+batch_size]

        feed_dict = {X: train_batch_data, Y:train_batch_labels, phase_train:False}
            
        # ------------
        # Compute real Hv (v is fixed)
        # ------------
        grads1 = sess.run(grads, feed_dict)
        sess.run(update_direction_vars_ops)
        grads2 = sess.run(grads, feed_dict)      
        sess.run(update_original_vars_ops) 
        
        # compute Hessian-vector-product (Hv)
        Hv_t = compute_hv(grads1, grads2)
        
        # compute cumulative sum
        mean_hv_value = my_utils.cumulative_sum(mean_hv_value, Hv_t)
          
    # compute real mean     
    mean_hv_value = [hv_/num_batches for hv_ in mean_hv_value]             
    return mean_hv_value

def update_v0(u, hv):
    new_v = []
    for u_, hv_ in zip(u, hv):
         v = u_ - eta*hv_
         new_v.append(v)
    return new_v

def update_v1(u, v, ul):
    new_v = []
    for u_, v_, ul_ in zip(u, v, ul): 
        v_ = v_ - a*u_ - b*ul_
        new_v.append(v_)
    return new_v

def update_v3(Q, U):
    new_ev = []
    for q in Q:
        ev = np.matmul(q, U)
        new_ev.append(ev)
    return new_ev


print ('-'*20 + '\nEvaluations on min-eig-val...\n' + '-'*20)

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
Hv_placeholder = [tf.placeholder(dtype=tf.float32, shape=w.shape.as_list()) for w in W]

# Create a saver.
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5000)
grads = tf.gradients(loss_op, W)


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

init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
sess.run(init)
 

model_iter = '-%s.'%str(FLAGS.model_iter) if FLAGS.model_iter != 99 else 'data-'
checkpoint_dir = './models_%s-%d_v%d_%s_L2_%s/C%d/exp_%d/'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2), FLAGS.dataset, FLAGS.random_seed)

model_dir = [checkpoint_dir + f.split('.data')[0] for f in listdir(checkpoint_dir) if 'data-' in f and model_iter in f]
print (model_dir)
for model__ in model_dir:
    
    load_iter = int(model__.split('-')[-1])
    if load_iter not in [20000, 40000, 60000, 80000]:
      continue 
  
    # Restores from checkpoint with relative path.
    saver.restore(sess, model__)

    k = 60 # number of epochs to go through dataset
    eta = initial_learning_rate #recommended values: 1e-3, 1e-4, 1e-5
    r = 0.001
    shapes = [w.shape.as_list() for w in W]
    
    np.random.seed(123)
    u0 = []
    for s in shapes:
        if len(s) == 1:
            u0.append( np.random.rand(s[0]))
        if len(s) == 2:
            u0.append(np.random.rand(s[0], s[1]))
        if len(s) == 4:
            u0.append(np.random.rand(s[0], s[1], s[2], s[3]))
                
    norm_term = np.sqrt(np.sum([np.square(norm(u)) for u in u0]))
    
    if True:
        
        print ('LANCZOS METHOD !')
        u =  [r*u.copy() / norm_term for u in u0] 
        ul = [zeros(u.shape) for u in u0 ]
        
        A =  zeros(k+1)  
        B =  zeros(k)
        b = 0
        
        #v = u - eta*Hv(u)
        v = update_v0(u, Hv(u))
        a = np.sum([np.inner(v_.flatten(), u_.flatten()) for v_, u_ in zip(v, u)] ) # a = v.T*u : sum over all layers 
        
        A[0] = a  # matlab: index starts with 1 but start with 0 for python
        Q = [v_.flatten() for v_ in v] # for all layers 
        obj = []
        
        count_real_min_eigen_val = []
        count_itertion = []
        for i in range(k):
    
            #v = v - a*u - b*ul
            v = update_v1(u, v, ul)
    
            #b = norm(v)
            b = np.sqrt(np.sum([np.square(norm(v_)) for v_ in v])) 
            
            ul = u # remember previous results for all layers 
            u = [v_/b for v_ in v]
            
            #v = u - eta*Hv(u)
            v = update_v0(u, Hv(u))
    
            #a = v.T*u
            a = np.sum([np.inner(v_.flatten(), hv.flatten()) for v_, hv in zip(v, u)] ) # a = v.T*u : sum over all layers 
        
            A[i+1] = a
            Q =  [hstack((q_.reshape(q_.shape[0], -1), u_.flatten().reshape(q_.shape[0], -1))) for q_, u_ in zip(Q, u)] # flatten u to vector
    
            B[i] = b
            T = diag(A[0:i+2]) + diag(B[0:i+1], 1) + diag(B[0:i+1], -1)
            
            if i > 2:
    	  
                (S, U) = eigs(T, k=1, which='LM')
                U = U.real
                
                # ev=Q*U[:,1] #a.real
                ev = update_v3(Q, U) 
                ev_reshape = [ev_.real.astype(np.float32).reshape(shape_) for ev_, shape_ in zip(ev, shapes)]
                norm__ = np.sum([np.square(norm(ev_))  for ev_ in ev_reshape])
                obj= np.sum([np.inner(ev_.flatten(), hv.flatten())/norm__ for ev_, hv in zip(ev, Hv(ev_reshape))] )
                print ('\rlamda[%d]:,'%i, obj.real)
                count_real_min_eigen_val.append(obj.real)
                df = pd.DataFrame(data={'real_lamda_%d_v3'%load_iter:count_real_min_eigen_val})
                if not os.path.exists('./logs_eval/'):
                    os.makedirs('./logs_eval/')      
                df.to_csv('./logs_eval/min_eig_val_%s-%d_v%d_%s_C%d_exp_%d_lr_%s_use_L2_%s_batch_%d_i-%d.csv'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, FLAGS.dataset, FLAGS.random_seed, str(eta), str(FLAGS.use_L2), batch_size, load_iter))   
 
