import tensorflow as tf
import numpy as np
import re


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def cross_entropy_loss_with_l2(logits, labels, W=[], weight_decay=0.0005, use_L2=True):
    labels = tf.cast(labels, tf.int64)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    print ('use_L2: [{}]'.format(use_L2))
    if use_L2:
        var_list_no_bias = [var for var in W if len(var.get_shape().as_list()) != 1] # no bias added
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for idx, var in enumerate(var_list_no_bias)])
        loss_op = loss_op + l2_loss*weight_decay
    return loss_op

def create_placeholders(params):
    weights_holder = []
    for idx, item in enumerate(params):
        tensor_shape = item.get_shape().as_list()
        w = tf.placeholder(tf.float32, tensor_shape, name='W_avg_%d'%idx)
        weights_holder.append(w)
    return weights_holder

def compute_average_weights(W_mean, W_t, t):
    new_W_mean = []
    for idx, w_bar in enumerate(W_mean):
        w_bar_new =  (w_bar*t + W_t[idx])/(t + 1)  # make sure t0 = 0, otherwise, we need to chaneg t to t-1
        new_W_mean.append(w_bar_new)
    return new_W_mean


def cumulative_sum(W_mean, W_t):
        new_W_mean = []
        for idx, w_bar in enumerate(W_mean):
            new_w_bar = w_bar + W_t[idx]
            new_W_mean.append(new_w_bar)
        return new_W_mean
    

def learning_rate_decay(version, learning_rate, num_iters=90000):
    print ('Learning_decay: [v%d]'%version)
    if version == 0:
        lr_s = [learning_rate/t for t in range(1, num_iters+1)]
        T_s = [t for t in range(num_iters)]
        num_S = len(T_s)
    if version == 1:
        lr_s = [learning_rate/np.sqrt(t) for t in range(1, num_iters+1)]
        T_s = [t for t in range(num_iters)]
        num_S = len(T_s)
    if version == 2:
        lr_s = [learning_rate]*40000 + [learning_rate/10]*20000 + [learning_rate/100]*20000 +  [learning_rate/100]*2000000
        T_s = [t for t in range(num_iters)]
        num_S = len(T_s)
    if version in [3, 4]:
        lr_s = [learning_rate]*40000 + [learning_rate/10]*20000 + [learning_rate/100]*20000 +  [learning_rate/100]*2000000
        T_s = [40000, 20000, 20000, num_iters-(4+2+2)*10000]
        num_S = len(T_s)
        print ('num_stages: [%d]'%num_S)
    return lr_s, T_s, num_S


