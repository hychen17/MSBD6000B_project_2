
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
        path = '/Users/hchenbo/Desktop/proj2/data'+ path[1:].strip()
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


train_img, train_label = load_input('/Users/hchenbo/Desktop/proj2/data/train.txt')
print(train_img.shape)
print(train_label.shape)
#%%

val_img, val_label = load_input('/Users/hchenbo/Desktop/proj2/data/val.txt')
print(val_img.shape)
print(val_label.shape)

#%%

batch_size = 64
num_feature_map = 64
fc_node = 1024

batches = []


def train_minibatch(train_img,train_label,batch_size):
    batches = []
    num_of_minibatch = int(2568/batch_size)
    indices = np.arange(2568)
    np.random.shuffle(indices)
    for i in range(num_of_minibatch):
        ind = indices[(i*batch_size):((i+1)*batch_size)]
        feature_batch = train_img[ind,:,:,:]
        label_batch = train_label[ind,:]
        batches.append([feature_batch,label_batch])
        
    return batches    

def test_minibatch(test_img,test_label,batch_size):
    batches = []
    indices = np.arange(550)
    num_of_minibatch = int(550/batch_size)
    np.random.shuffle(indices)
    for i in range(num_of_minibatch):
        ind = indices[(i*batch_size):((i+1)*batch_size)]
        feature_batch = test_img[ind,:,:,:]
        label_batch = test_label[ind,:]
        batches.append([feature_batch,label_batch])
        
    return batches


#for i in range(40):
#    if i != 40:
#        feature_batch = train_img[(i*batch_size):((i+1)*batch_size),:,:,:]
#        label_batch = train_label[(i*batch_size):((i+1)*batch_size),:]
#        batches.append([feature_batch,label_batch])
#    else:
#        feature_batch = train_img[(i*batch_size):,:,:,:]
#        label_batch = train_label[(i*batch_size):,:]
#        batches.append([feature_batch,label_batch])


#def minibatches(x, y, batch_size=1, shuffle=False):
#    assert len(x) == len(y)
#    if shuffle:
#        index = np.arange(len(y))
#        np.random.shuffle(index)
#    for start_index in range(0, len(x) - batch_size + 1, batch_size):
#        if shuffle:
#            local_index = index[start_index:start_index + batch_size]
#        else:
#            local_index = slice(start_index, start_index + batch_size)
#        yield x[local_index], y[local_index]
#    if(start_index+batch_size<len(y)-1):
#        if shuffle:
#            local_index=index[start_index + batch_size:len(y)]
#        else:
#            local_index = slice(start_index + batch_size,len(y))
#        yield x[local_index], y[local_index]
    
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



# [batch_size * 231 * 231 * 3]
x_image = tf.placeholder(tf.float32, [batch_size,IMG_DIM,IMG_DIM,3])


W_conv1 = weight_variable([3, 3, 3, num_feature_map])
b_conv1 = bias_variable([num_feature_map])

# (64, 231, 231, 64)
h_conv1 = tf.nn.relu(conv2d_1(x_image, W_conv1) + b_conv1)
print('h_conv1: ',str(h_conv1.get_shape()))

# (64, 116, 116, 64)
h_pool1 = max_pool_2x2(h_conv1)
print('h_pool1: ',str(h_pool1.get_shape()))

W_conv2 = weight_variable([3, 3, num_feature_map, num_feature_map])
b_conv2 = bias_variable([num_feature_map])

# (64, 116, 116, 64)
h_conv2 = tf.nn.relu(conv2d_2(h_pool1, W_conv2) + b_conv2)
print('h_conv2: ',str(h_conv2.get_shape()))

# (64, 58, 58, 64)
h_pool2 = max_pool_2x2(h_conv2)
print('h_pool2: ',str(h_pool2.get_shape()))


W_conv3 = weight_variable([3, 3, num_feature_map, num_feature_map])
b_conv3 = bias_variable([num_feature_map])

# (64, 58, 58, 64)
h_conv3 = tf.nn.relu(conv2d_3(h_pool2, W_conv3) + b_conv3)
print('h_conv3: ',str(h_conv3.get_shape()))

# (64, 29, 29, 64)
h_pool3 = max_pool_2x2(h_conv3)
print('h_pool3: ',str(h_pool3.get_shape()))

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
dim = int(h_pool3.get_shape()[1]*h_pool3.get_shape()[2]*h_pool3.get_shape()[3])
W_fc1 = weight_variable([dim, fc_node])
b_fc1 = bias_variable([fc_node])

#(64, 53824)
h_pool3_flat = tf.reshape(h_pool3, [-1, dim])
print('h_pool3_flat: ',str(h_pool3_flat.get_shape()))
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
print('h_fc1: ',str(h_fc1.get_shape()))

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#(64, 1024)
# full connect 2
W_fc2 = weight_variable([fc_node, 5])
b_fc2 = bias_variable([5])

#(64, 5)
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print('y_conv: ',str(y_conv.get_shape()))
y_ = tf.placeholder(tf.float32, [None, 5])

#%%

# model training

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

start_time = time.time()
for epoch in range(6):
    train_batches = train_minibatch(train_img,train_label,64)
    test_batches = test_minibatch(val_img,val_label,64)
    i = 0
    for batch in train_batches:
            i += 1
            sess.run(train_step, feed_dict = {x_image: batch[0], y_: batch[1], keep_prob: 0.8})
            train_accuacy = sess.run(accuracy, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 0.8})
            print("Epoch: ", str(epoch)," step %d, training accuracy %g"%(i, train_accuacy))
            print('total time elapsed: ' + str(time.time()-start_time))
                # accuacy on test
    test_acc = 0
    i = 0
    for batch in test_batches:   
        i += 1
        test_accuacy = sess.run(accuracy, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 0.8})
        test_acc += test_accuacy
    print("Epoch: ", str(epoch)," testing accuracy: ", str(test_acc/7))
    print('total time elapsed: ' + str(time.time()-start_time))

save_path = saver.save(sess, "/Users/hchenbo/Desktop/proj2/model/model.ckpt")
print("Model saved in file: %s" % save_path)

#%%
test_acc = 0
test_batches = test_minibatch(val_img,val_label,64)
i = 0
for batch in test_batches:   
    i += 1
    test_accuacy = sess.run(accuracy, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 0.8})
    test_acc += test_accuacy
print("Epoch: ", str(epoch)," testing accuracy: ", str(test_acc/8))
print('total time elapsed: ' + str(time.time()-start_time))
#%%
train_batches = train_minibatch(train_img,train_label,64)
train_acc = 0
i = 0
for batch in train_batches:
    i += 1
    sess.run(train_step, feed_dict = {x_image: batch[0], y_: batch[1], keep_prob: 0.8})
    train_accuacy = sess.run(accuracy, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 0.8})
    train_acc += train_accuacy
print("Epoch: ", str(epoch)," training accuracy: ", str(train_acc/40))
print('total time elapsed: ' + str(time.time()-start_time))

#%%
import numpy as np
import math
import os 
import random
import tensorflow as tf
import cv2


IMG_DIM = 231
def load_test_input(imges_path):
    test_img_path = []

    
    with open(imges_path,'r') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            test_img_path.append(line)           
     
    test_img = []
    
    for ind,path in enumerate(test_img_path):
        path = '/Users/hchenbo/Desktop/proj2/data'+ path[1:].strip()
        if path.endswith('\n'):
            path = path[:-2]
        #img = cv2.imread(path)
        #img = cv2.resize(img,(IMG_DIM, IMG_DIM))
        #img = img.tolist()
        #resized_image = (cv2.imread(path)).tolist()
        test_img.append(list(cv2.resize(cv2.imread(path),(IMG_DIM,IMG_DIM))))
        
        if len(test_img)%50 == 0:
                print(len(test_img))
                
    test_img = np.array(test_img)

    return test_img


test_img = load_test_input('/Users/hchenbo/Desktop/proj2/data/test.txt')
print(test_img.shape)

#%%
batch_size = 64
batches = []
for i in range(9):
    if i != 8:
        feature_batch = test_img[(i*batch_size):((i+1)*batch_size),:,:,:]
        batches.append([feature_batch])
    else:
        feature_batch = test_img[(i*batch_size):,:,:,:]
        for i in range(25):
            feature_batch = np.vstack((feature_batch,test_img[550:551,:,:,:]))
        batches.append([feature_batch])
    
print(len(batches))

#%%

import csv

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "/Users/hchenbo/Desktop/proj2/model/model.ckpt")
predition = []
for i in range(9):
    result = sess.run(y_conv, feed_dict={x_image: batches[i][0], keep_prob: 0.8})
    predition += np.argmax(result,1).tolist()
    
with open('/Users/hchenbo/Desktop/proj2/model/project2_20451451.csv','w', newline='') as f:
    writer = csv.writer(f)
    count = 0
    for v in predition:
        count += 1
        writer.writerows([str(v)])
        if count == 551:
            break
#%%
