
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:54:12 2017

@author: haoyuan
"""
#%%

import tensorflow as tf
import numpy as np
import math
import time
import cv2
from datetime import datetime
import os
import re
import sys

# load data-sets
# load data-sets
IMG_DIM = 231
def load_input(imges_path):
    train_img_path = []

    train_img_label = []
    
    with open(imges_path,'r') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            train_img_path.append(line.split(' ')[0])
            if line.split(' ')[1] == '0':
                train_img_label.append([0,0,0,0,1])
            elif line.split(' ')[1] == '1':  
                train_img_label.append([0,0,0,1,0])
            elif line.split(' ')[1] == '2':
                train_img_label.append([0,0,3,0,0])
            elif line.split(' ')[1] == '3':
                train_img_label.append([0,4,0,0,0])
            elif line.split(' ')[1] == '4':
                train_img_label.append([5,0,0,0,0])
            
     
    train_img = []
    
    for ind,path in enumerate(train_img_path):
        path = 'E:\\ust\\6000B\\proj2\\data'+ path[1:].strip()
        if path.endswith('\n'):
            path = path[:-2]
        #img = cv2.imread(path)
        #img = cv2.resize(img,(IMG_DIM, IMG_DIM))
        #img = img.tolist()
        #resized_image = (cv2.imread(path)).tolist()
        train_img.append(list(cv2.resize(cv2.imread(path),(IMG_DIM,IMG_DIM))))
        
        if len(train_img)%50 == 0:
                print(len(train_img))
                
    train_img = np.array(train_img)
    train_img_label = np.array(train_img_label)

    return train_img, train_img_label


train_img, train_label = load_input('E:\\ust\\6000B\\proj2\\data\\train.txt')
print(train_img.shape)
print(train_label.shape)
#%%

val_img, val_label = load_input('E:\\ust\\6000B\\proj2\\data\\val.txt')
print(val_img.shape)
print(val_label.shape)

#%%

batch_size = 64
num_feature_map = 64
fc_node = 1024

batches = []
for i in range(40):
    if i != 40:
        feature_batch = train_img[(i*batch_size):((i+1)*batch_size),:,:,:]
        label_batch = train_label[(i*batch_size):((i+1)*batch_size),:]
        batches.append([feature_batch,label_batch])
    else:
        feature_batch = train_img[(i*batch_size):,:,:,:]
        label_batch = train_label[(i*batch_size):,:]
        batches.append([feature_batch,label_batch])
        
#%%

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d_1(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_3(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_33x33(x):
    return tf.nn.max_pool(x, ksize=[1, 33, 33, 1], strides=[1, 33, 33, 1], padding='SAME')

#%%



# [batch_size * 297 * 297 * 3]
x_image = tf.placeholder(tf.float32, [batch_size,IMG_DIM,IMG_DIM,3])
#x_image = tf.reshape(tf.float32, [batch_size,297,297,3])

W_conv1 = weight_variable([3, 3, 3, num_feature_map])
b_conv1 = bias_variable([num_feature_map])

# [batch_size * 99 * 99 * 9]
h_conv1 = tf.nn.relu(conv2d_1(x_image, W_conv1) + b_conv1)
print('h_conv1: ',str(h_conv1.shape))

# [batch_size * 33 * 33 * 9]
h_pool1 = max_pool_2x2(h_conv1)
print('h_pool1: ',str(h_pool1.shape))

W_conv2 = weight_variable([3, 3, num_feature_map, num_feature_map])
b_conv2 = bias_variable([num_feature_map])

# [batch_size * 16 * 16 * 9]
h_conv2 = tf.nn.relu(conv2d_2(h_pool1, W_conv2) + b_conv2)
print('h_conv2: ',str(h_conv2.shape))

# [batch_size * 8 * 8 * 9]
h_pool2 = max_pool_2x2(h_conv2)
print('h_pool2: ',str(h_pool2.shape))


W_conv3 = weight_variable([3, 3, num_feature_map, num_feature_map])
b_conv3 = bias_variable([num_feature_map])

# [batch_size * 16 * 16 * 9]
h_conv3 = tf.nn.relu(conv2d_3(h_pool2, W_conv3) + b_conv3)
print('h_conv3: ',str(h_conv3.shape))

# [batch_size * 8 * 8 * 9]
h_pool3 = max_pool_2x2(h_conv3)
print('h_pool3: ',str(h_pool3.shape))

#W_conv4 = weight_variable([3, 3, num_feature_map, num_feature_map])
#b_conv4 = bias_variable([num_feature_map])
#
## [batch_size * 16 * 16 * 9]
#h_conv4 = tf.nn.relu(conv2d_3(h_pool3, W_conv4) + b_conv4)
#print('h_conv4: ',str(h_conv4.shape))
#
## [batch_size * 8 * 8 * 9]
#h_pool4 = max_pool_2x2(h_conv4)
#print('h_pool4: ',str(h_pool4.shape))

# full connect 1
dim = int(h_pool3.shape[1]*h_pool3.shape[2]*h_pool3.shape[3])
W_fc1 = weight_variable([dim, fc_node])
b_fc1 = bias_variable([fc_node])

h_pool3_flat = tf.reshape(h_pool3, [-1, dim])
print('h_pool3_flat: ',str(h_pool3_flat.shape))
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
print('h_fc1: ',str(h_fc1.shape))

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# full connect 2
W_fc2 = weight_variable([fc_node, 5])
b_fc2 = bias_variable([5])


y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print('y_conv: ',str(y_conv.shape))
y_ = tf.placeholder(tf.float32, [None, 5])

#%%

# model training

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

start_time = time.time()
for epoch in range(40):
    for i in range(40):
        with tf.Graph().as_default():    
            sess.run(train_step, feed_dict = {x_image: batches[i][0], y_: batches[i][1], keep_prob: 1})
            train_accuacy = sess.run(accuracy, feed_dict={x_image: batches[i][0], y_: batches[i][1], keep_prob: 0.8})
            print("Epoch: ", str(epoch)," step %d, training accuracy %g"%(i, train_accuacy))
            print('total time elapsed: ' + str(time.time()-start_time))
                # accuacy on test
    test_acc = 0
    for i in range(7):
        with tf.Graph().as_default():    
                test_accuacy = sess.run(accuracy, feed_dict={x_image: val_img[(i*64):((i+1)*64),:,:,:], y_: val_label[(i*64):((i+1)*64),:], keep_prob: 0.8})
                test_acc += test_accuacy
    print("testing accuracy %g"%(test_accuacy/7)
    print('total time elapsed: ' + str(time.time()-start_time))

save_path = saver.save(sess, "E:\\ust\\6000B\\proj2\\model\\model.ckpt")
print("Model saved in file: %s" % save_path)

#%%