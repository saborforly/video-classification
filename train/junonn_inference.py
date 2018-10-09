from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

TOWER_NAME = 'tower'
FLAGS = tf.app.flags.FLAGS
#tf.contrib.layers.xavier_initializer_conv2d()
#tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
def conv_op(input_op,kernel_shape,stride,name):
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w", shape=kernel_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())        
        conv=tf.nn.conv2d(input_op,kernel,strides=stride,padding="SAME")
        
        
        n_out=kernel_shape[-1]
        #biases=tf.Variable(tf.constant(0.0,shape=[n_out],dtype=tf.float32),trainable=True,name='b')
        biases=tf.get_variable(scope+"b", shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        z=tf.nn.bias_add(conv, biases)
        activation=tf.nn.relu(z,name=scope+'relu')
        return activation
        
def fc_op(input_op,n_out,wd,name):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w", shape=[n_in,n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        #add L2 regularizer to w
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel),wd, name=scope+'weight_loss')
        tf.add_to_collection('losses', weight_decay)                
        
        biases=tf.get_variable(scope+"b", shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        activation=tf.nn.relu_layer(input_op, kernel, biases, name=scope+'relu')
        #activation = tf.nn.relu(tf.matmul(input_op, kernel) + biases, name=scope+'relu')
        return activation
    
def mpool_op(input_op,pool_shape,stride,name):
    return tf.nn.max_pool(input_op, ksize=pool_shape, strides=stride, padding='SAME',  
                       name=name)

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images):
    conv1=conv_op(images, [5,5,2,64], [1,1,1,1],name='conv1')  
    pool1=mpool_op(conv1, [1,3,3,1], [1,2,2,1], name='pool1')
        
    conv2=conv_op(pool1, [5,5,64,64], [1,1,1,1],name='conv2')     
    pool2=mpool_op(conv2, [1,3,3,1], [1,2,2,1], name='pool2')
    
    '''  
    conv3=conv_op(pool2, [3,3,192,384], [1,1,1,1],name='conv3_1')
    conv4=conv_op(conv3, [3,3,384,256], [1,1,1,1],name='conv3_2')
    conv5=conv_op(conv4, [3,3,256,256], [1,1,1,1],name='conv3_3')
    pool5=mpool_op(conv5, [1,3,3,1], [1,2,2,1], name='pool3')
    '''
        
        
        
   
    resh1 = tf.reshape(pool2, [FLAGS.batch_size, -1])
    fc6=fc_op(resh1, 384,0.0004,name='fc6')
       
    fc7=fc_op(fc6,192,0.0004,name='fc7')

    fc8=fc_op(fc7, 2,0.0004,name='fc8')


    
    
        
        
    _activation_summary(conv1)
    _activation_summary(conv2)
    _activation_summary(fc8)

    return fc8


