import tensorflow as tf
import numpy as np


def _batch_norm(input_layer, phase_train):
    bn_layer = tf.layers.batch_normalization(input_layer, training=phase_train, fused=None, momentum=0.99, epsilon=1e-3)
    return bn_layer

def convnet_inference(x, num_classes, num_layers=3,  activations='elu', phase_train=False):

    with tf.name_scope('ConvNet'): 
        # Convolution Layer 1
        conv1 = tf.layers.conv2d(x, 32, 5, activation=None, use_bias=False)
        conv1_bn = _batch_norm(conv1, phase_train)
        conv1_out = tf.nn.elu(conv1_bn) if activations=='elu' else tf.nn.relu(conv1_bn)
        conv1 = tf.layers.average_pooling2d(conv1_out, 2, 2)
       
        # Convolution Layer 2
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=None, use_bias=False)
        conv2_bn = _batch_norm(conv2, phase_train)
        conv2_out = tf.nn.elu(conv2_bn) if activations=='elu' else tf.nn.relu(conv2_bn)
        conv2 = tf.layers.average_pooling2d(conv2_out, 2, 2)
        
        # Convolution Layer 3
        conv3 = tf.layers.conv2d(conv2, 64, 3, activation=None, use_bias=False)
        conv3_bn = _batch_norm(conv3, phase_train)
        conv3_out = tf.nn.elu(conv3_bn) if activations=='elu' else tf.nn.relu(conv3_bn)
        conv3 = tf.layers.average_pooling2d(conv3_out, 2, 2)     

        # fully connected layer
        fc1 = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(fc1, 1024)
        out = tf.layers.dense(fc1, num_classes)
    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    print ('bulid ConvNet: %d'%( num_layers))
    print ('parameters: [%d]'%(num_params))
    print ('Activation:[%s]'%activations)
    return out

