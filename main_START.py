# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from datetime import datetime
from convnet import convnet_inference
from resnet_model import resnet_inference
import pandas as pd
import os 
import cifar_input as cifar_data
import my_utils
import optimzer as opt
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.reset_default_graph()

try:
    # Try/Except: for debug purpose in windows
    FLAGS.activation
except:
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('model', 'resnet', '''which model to train: resnet or convnet''')
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''random seed for initialization''')
    tf.app.flags.DEFINE_integer('train_batch_size', 128, '''batch_size''')
    tf.app.flags.DEFINE_float('lr', 0.1, '''learning rate to train the models''')
    tf.app.flags.DEFINE_integer('dataset', 10, '''dataset to evalute: 10 or 100''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_integer('gamma', 5000, '''gamma for START, leave this any values if train other models''')
    tf.app.flags.DEFINE_integer('num_iters', 230*400, '''total number of iterations to run''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data''')
    tf.app.flags.DEFINE_boolean('is_save_model', False, '''whether to save model or not ''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer''')
    tf.app.flags.DEFINE_integer('version', 3, '''[3:SGDv2, 4:SGDv3]''')

tf.set_random_seed(FLAGS.random_seed)

def eval_once(images, labels):
    num_batches = images.shape[0]//batch_size
    accuracy_mean = []
    loss_mean = []
    for step in range(num_batches):
        offset = step * batch_size
        vali_data_batch = images[offset:offset+batch_size]
        vali_label_batch = labels[offset:offset+batch_size]
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: vali_data_batch, Y: vali_label_batch, phase_train:False})
        accuracy_mean.append(acc)
        loss_mean.append(loss)
    return np.mean(loss_mean), np.mean(accuracy_mean)
        
# Import CIFAR data
(train_data, train_labels), (test_data, test_labels) = cifar_data.load_data(FLAGS.dataset, FLAGS.is_tune)

# Training Parameters
initial_learning_rate = FLAGS.lr
num_iters = FLAGS.num_iters
batch_size = FLAGS.train_batch_size
inference = resnet_inference if FLAGS.model == 'resnet'else convnet_inference

# Network Parameters
configs = '_T_%d_B_%d_lr_%.4f_g_%d_L2_%s_is_tune_%s_%s-%d[v%d-%s]-C%d_seed_%d'%(FLAGS.num_iters, batch_size, initial_learning_rate, FLAGS.gamma, FLAGS.use_L2, str(FLAGS.is_tune), FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, FLAGS.dataset, FLAGS.random_seed)

# create tf Graph input
X = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
Y = tf.placeholder(tf.float32, [batch_size,])
lr = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

# Construct model
logits = inference(X, num_classes=FLAGS.dataset, num_layers=FLAGS.resnet_layers, activations=FLAGS.activation, phase_train=phase_train) # when resnet you need to pass number of layers 
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
W = [var for var in tf.trainable_variables ()]
loss_op = my_utils.cross_entropy_loss_with_l2(logits, Y, W, use_L2=FLAGS.use_L2)
W_avg = my_utils.create_placeholders(W)
train_op = opt.START(loss_op, W, W_avg, lr=lr, gamma=FLAGS.gamma)
lr_s, T_s, num_S = my_utils.learning_rate_decay(FLAGS.version, initial_learning_rate, num_iters=num_iters)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# store models
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5000)
init = tf.global_variables_initializer()
global_iteration = 0

# Start training
with tf.Session() as sess:  
   
    # Run the initializer
    sess.run(init)
    train_loss_list, test_loss_list, train_acc_list, test_acc_list, lr_list = [], [], [], [], []
    print ('Check W0: [%.3f]'%(sess.run(W))[0].sum())
    print ('\nStart training...')
    W_mean = sess.run(W)
    W_t = sess.run(W)   
    
    for s in range(num_S):
        
        var_mean_ = W_mean if FLAGS.version == 4 else W_t
            
        for iter_ in range(T_s[s]):
            
            learning_rate = lr_s[global_iteration] 
            batch_x, batch_y = cifar_data.generate_augment_train_batch(train_data, train_labels, batch_size, FLAGS.is_tune)
            
            feed_dict = {X: batch_x, Y: batch_y, lr: learning_rate, phase_train:True}
            for idx, w_ in enumerate(W_avg):
                feed_dict[w_] = var_mean_[idx]

            sess.run(train_op, feed_dict=feed_dict)

            # needed for START
            W_t = sess.run(W) # return value is a list 
            if FLAGS.version == 4:
                W_mean = my_utils.compute_average_weights(W_mean, W_t, iter_) # compute average weights gradually 
        
            if iter_ % 400 == 0:
                
                # evaluate as average /current
                if FLAGS.version == 4:
                    update_mean_ops = [var.assign(W_mean[idx]) for idx, var in enumerate(W) if len(var.get_shape().as_list()) != 1] 
                    sess.run(update_mean_ops)
                
                train_loss, train_acc = eval_once(train_data, train_labels)
                test_loss, test_acc = eval_once(test_data, test_labels)
                
                if FLAGS.version == 4:
                    update_current_ops = [var.assign(W_t[idx]) for idx, var in enumerate(W) if len(var.get_shape().as_list()) != 1]
                    sess.run(update_current_ops)  
                
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                lr_list.append(learning_rate)
                print("%s: [%d]-[%d]: lr:%.5f, Train_Loss: %.5f, Test_loss: %.5f, Train_acc:, %.5f, Test_acc:, %.5f"%(datetime.now(), s, iter_, learning_rate, train_loss, test_loss, train_acc, test_acc))
                        
                df = pd.DataFrame(data={'Train_loss'+configs:train_loss_list, 'Test_loss'+configs:test_loss_list, 'Train_acc'+configs:train_acc_list, 'Test_acc'+configs: test_acc_list})
                df.to_csv('./logs/[v%d]'%FLAGS.version + configs + '.csv')
                
                if not FLAGS.is_tune and FLAGS.is_save_model:
                    save_dir = './models_%s-%d_v%d_%s_L2_%s/C%d/exp_%d/'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2), FLAGS.dataset, FLAGS.random_seed)
                    checkpoint_path = os.path.join(save_dir, 'model_%s-%d_v%d_cifar%d_%s.ckpt'%(FLAGS.model, FLAGS.resnet_layers,  FLAGS.version, FLAGS.dataset, FLAGS.activation))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)                  
                    saver.save(sess, checkpoint_path, global_step=global_iteration, write_meta_graph=False)
                    
            # update counter 
            global_iteration += 1      
                   
        if FLAGS.version ==4:
            # update average W to model when stage ends
            update_weights_ops = [var.assign(W_mean[idx]) for idx, var in enumerate(W) if len(var.get_shape().as_list()) != 1]
            sess.run(update_weights_ops) 