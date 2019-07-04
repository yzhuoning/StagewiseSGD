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
    tf.app.flags.DEFINE_integer('dataset', 100, '''dataset to evalute: 10 or 100''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20, '''number of layers to use in ResNet: 56 or 20; if convnet, make it to 3''')
    tf.app.flags.DEFINE_integer('num_iters', 230*400, '''total number of iterations to train the model''')
    tf.app.flags.DEFINE_boolean('is_tune', False, '''if True, split train dataset (50K) into 45K, 5K as train/validation data''')
    tf.app.flags.DEFINE_boolean('is_save_model', False, '''whether to save model or not ''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer''')
    tf.app.flags.DEFINE_integer('version', 2, '''[0: c/t, 1: c/sqrt(t),  2:SGDv1]''')

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
configs = '_T_%d_B_%d_lr_%.4f_L2_%s_is_tune_%s_%s-%d[v%d-%s]-C%d_seed_%d'%(FLAGS.num_iters, batch_size, initial_learning_rate,  FLAGS.use_L2, str(FLAGS.is_tune), FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, FLAGS.dataset, FLAGS.random_seed)

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
train_op = opt.SGD(loss_op, lr=lr)
lr_s, _, _ = my_utils.learning_rate_decay(FLAGS.version, initial_learning_rate, num_iters=num_iters)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# store models
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5000)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:  
   
    # Run the initializer
    sess.run(init)
    train_loss_list, test_loss_list, train_acc_list, test_acc_list, lr_list = [], [], [], [], []
    print ('Check W0: [%.3f]'%(sess.run(W))[0].sum())
    print ('\nStart training...')
    
    for iter_ in range(num_iters):
  
        learning_rate = lr_s[iter_] 
        batch_x, batch_y = cifar_data.generate_augment_train_batch(train_data, train_labels, batch_size, FLAGS.is_tune)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, lr: learning_rate, phase_train:True})
               
        if iter_ % 400 == 0:
            # Calculate loss and accuracy over entire dataset
            train_loss, train_acc = eval_once(train_data, train_labels)
            test_loss, test_acc = eval_once(test_data, test_labels)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            lr_list.append(learning_rate)
            print("%s: [%d]: lr:%.5f, Train_Loss: %.5f, Test_loss: %.5f, Train_acc:, %.5f, Test_acc:, %.5f"%(datetime.now(), iter_, learning_rate, train_loss, test_loss, train_acc, test_acc))
                    
            df = pd.DataFrame(data={'Train_loss'+configs:train_loss_list, 'Test_loss'+configs:test_loss_list, 'Train_acc'+configs:train_acc_list, 'Test_acc'+configs: test_acc_list})
            df.to_csv('./logs/[v%d]'%FLAGS.version + configs + '.csv')
            
            if not FLAGS.is_tune and FLAGS.is_save_model:
                save_dir = './models_%s-%d_v%d_%s_L2_%s/C%d/exp_%d/'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2), FLAGS.dataset, FLAGS.random_seed)
                checkpoint_path = os.path.join(save_dir, 'model_%s-%d_v%d_cifar%d_%s.ckpt'%(FLAGS.model, FLAGS.resnet_layers, FLAGS.version, FLAGS.dataset, FLAGS.activation))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)                  
                if iter_ < 90400 or iter_> 199600:
                   saver.save(sess, checkpoint_path, global_step=iter_, write_meta_graph=False)
                 