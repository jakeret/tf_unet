# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tf_unet.util import combine_img_prediction
from tf_unet.util import crop_to_shape
import shutil

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def weight_variable_devonc(shape):
#     initial = 1.0/float(shape[0]*shape[1])
    #return tf.Variable(tf.add((np.ones(shape)*initial).astype(float32),tf.truncated_normal(shape, stddev=0.1)))
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride):
    x_shape = tf.shape(x)
    output_shape = tf.pack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2,output_shape):
    #x1.set_shape(input_shape)
#     x1_crop = tf.image.extract_glimpse(x1, [int(output_shape[1]), int(output_shape[2])], np.zeros([output_shape[0],2]), centered=True)
    offsets = tf.zeros(tf.pack([output_shape[0], 2]), dtype=tf.float32)
#     size = tf.to_int32(tf.pack([output_shape[1], output_shape[2]]))
    x2_shape = tf.shape(x2)
    size = tf.pack((x2_shape[1], x2_shape[2]))
    x1_crop = tf.image.extract_glimpse(x1, size=size, offsets=offsets, centered=True)
    #x1_crop = tf.image.extract_glimpse(x1, [tf.shape(x2)[1], tf.shape(x2)[2]],tf.shape(x2)[0])
    return tf.concat(3, [x1_crop, x2]) 

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.pack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))

def create_conv_net(x, keep_prob, channels, n_class, layers=2, chanel_root=32, field_of_view=3, max_pool_size=2):
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.pack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = {}
    deconv = {}
    h_convs = {}
    
    # down layers
    for layer in range(0, layers):
        features = 2**layer*chanel_root
        if layer == 0:
            w1 = weight_variable([field_of_view, field_of_view, channels, features])
        else:
            w1 = weight_variable([field_of_view, field_of_view, features//2, features])
            
        w2 = weight_variable([field_of_view, field_of_view, features, features])
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        if layer < layers-1:
            pools[layer] = max_pool(h_convs[layer], max_pool_size)
            in_node = pools[layer]
        
    in_node = h_convs[layers-1]
        
    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*chanel_root
        
        wd = weight_variable_devonc([max_pool_size, max_pool_size, features//2, features])
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, max_pool_size) + bd)
        h_deconv_concat = crop_and_concat(h_convs[layer], h_deconv, [batch_size])
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([field_of_view, field_of_view, features, features//2])
        w2 = weight_variable([field_of_view, field_of_view, features//2, features//2])
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

    # Output Map
    weight = weight_variable([1, 1, chanel_root, n_class])
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    
    for i, (c1, c2) in enumerate(convs):
        tf.image_summary('summary_conv_%02d_01'%i, get_image_summary(c1))
        tf.image_summary('summary_conv_%02d_02'%i, get_image_summary(c2))
        
    for k in sorted(pools.keys()):
        tf.image_summary('summary_pool_%02d'%k, get_image_summary(pools[k]))
    
    for k in sorted(deconv.keys()):
        tf.image_summary('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))
        
    return output_map

def create_conv_net2(x, keep_prob, channels, n_class, chanel_root = 32, field_of_view = 3, max_pool_size = 2):
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.pack([-1,nx,ny,channels]))
    batch_size = tf.shape(x_image)[0]
     
    # Convolution 1
    W_conv1 = weight_variable([field_of_view, field_of_view, channels, chanel_root])
    b_conv1 = bias_variable([chanel_root])
    conv_1 = conv2d(x_image, W_conv1,keep_prob)
    h_conv1 = tf.nn.relu(conv_1 + b_conv1)
     
    # Convolution 2
    W_conv2 = weight_variable([field_of_view, field_of_view, chanel_root, chanel_root])
    b_conv2 = bias_variable([chanel_root])
    conv_2 = conv2d(h_conv1, W_conv2,keep_prob)
    h_conv2 = tf.nn.relu(conv_2 + b_conv2)
     
    # Max Pool 1
    h_pool1 = max_pool(h_conv2, max_pool_size)
      
    # Convolution 3
    W_conv3 = weight_variable([field_of_view, field_of_view, chanel_root, 2*chanel_root])
    b_conv3 = bias_variable([2*chanel_root])
    conv_3 = conv2d(h_pool1, W_conv3,keep_prob)
    h_conv3 = tf.nn.relu(conv_3 + b_conv3)
      
    # Convolution 4
    W_conv4 = weight_variable([field_of_view, field_of_view, 2*chanel_root, 2*chanel_root])
    b_conv4 = bias_variable([2*chanel_root])
    conv_4 = conv2d(h_conv3, W_conv4,keep_prob)
    h_conv4 = tf.nn.relu(conv_4 + b_conv4)
      
#     # Max Pool 2
#     h_pool2 = max_pool(h_conv4, max_pool_size)
#      
#     # Convolution 5
#     W_conv5 = weight_variable([field_of_view, field_of_view, 2*chanel_root, 4*chanel_root])
#     b_conv5 = bias_variable([4*chanel_root])
#     conv_5 = conv2d(h_pool2, W_conv5,keep_prob)
#     h_conv5 = tf.nn.relu(conv_5 + b_conv5)
#      
#     # Convolution 6
#     W_conv6 = weight_variable([field_of_view, field_of_view, 4*chanel_root, 4*chanel_root])
#     b_conv6 = bias_variable([4*chanel_root])
#     conv_6 = conv2d(h_conv5, W_conv6,keep_prob)
#     h_conv6 = tf.nn.relu(conv_6 + b_conv6)
#      
#     # Max Pool 3
#     h_pool3 = max_pool(h_conv6, max_pool_size)
#      
#     # Convolution 7
#     W_conv7 = weight_variable([field_of_view, field_of_view, 4*chanel_root, 8*chanel_root])
#     b_conv7 = bias_variable([8*chanel_root])
#     conv_7 = conv2d(h_pool3, W_conv7,keep_prob)
#     h_conv7 = tf.nn.relu(conv_7 + b_conv7)
#      
#     # Convolution 8
#     W_conv8 = weight_variable([field_of_view, field_of_view, 8*chanel_root, 8*chanel_root])
#     b_conv8 = bias_variable([8*chanel_root])
#     conv_8 = conv2d(h_conv7, W_conv8,keep_prob)
#     h_conv8 = tf.nn.relu(conv_8 + b_conv8)
#      
#     # Max Pool 4
#     h_pool4 = max_pool(h_conv8, max_pool_size)
#      
#     # Convolution 9
#     W_conv9 = weight_variable([field_of_view, field_of_view, 8*chanel_root, 16*chanel_root])
#     b_conv9 = bias_variable([16*chanel_root])
#     conv_9 = conv2d(h_pool4, W_conv9,keep_prob)
#     h_conv9 = tf.nn.relu(conv_9 + b_conv9)
#      
#     # Convolution 10
#     W_conv10 = weight_variable([field_of_view, field_of_view, 16*chanel_root, 16*chanel_root])
#     b_conv10 = bias_variable([16*chanel_root])
#     conv_10 = conv2d(h_conv9, W_conv10,keep_prob)
#     h_conv10 = tf.nn.relu(conv_10 + b_conv10)
#      
#     # Deconvolution 1
#     W_deconv_1 = weight_variable_devonc([max_pool_size, max_pool_size, 8*chanel_root, 16*chanel_root])
#     b_deconv1 = bias_variable([8*chanel_root])
#     deconv_1 = deconv2d(h_conv10, W_deconv_1, max_pool_size)
#     h_deconv1 = tf.nn.relu(deconv_1 + b_deconv1)
#     h_deconv1_concat = crop_and_concat(h_conv8, h_deconv1, [batch_size, (((nx-180)/2+4)/2+4)/2+4, (((ny-180)/2+4)/2+4)/2+4, 8*chanel_root])
# #     
# #     # Convolution 11
# #     offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
# #     size = tf.to_int32(tf.pack([(((nx-192)/2+4)/2+4)/2+4, (((ny-192)/2+4)/2+4)/2+4]))
# #     h_conv11 = tf.image.extract_glimpse(h_conv8, size=size, offsets=offsets, centered=True)
#     W_conv11 = weight_variable([field_of_view, field_of_view, 16*chanel_root, 8*chanel_root])
#     b_conv11 = bias_variable([8*chanel_root])
#     h_conv11 = tf.nn.relu(conv2d(h_deconv1_concat, W_conv11,keep_prob) + b_conv11)
# #     
#     # Convolution 12
#     W_conv12 = weight_variable([field_of_view, field_of_view, 8*chanel_root, 8*chanel_root])
#     b_conv12 = bias_variable([8*chanel_root])
#     h_conv12 = tf.nn.relu(conv2d(h_conv11, W_conv12,keep_prob) + b_conv12)
#      
#     # Deconvolution 2
#     W_deconv_2 = weight_variable_devonc([max_pool_size, max_pool_size, 4*chanel_root, 8*chanel_root])
#     b_deconv2 = bias_variable([4*chanel_root])
#     h_deconv2 = tf.nn.relu(deconv2d(h_conv12, W_deconv_2, max_pool_size) + b_deconv2)
#     h_deconv2_concat = crop_and_concat(h_conv6,h_deconv2,[batch_size,((nx-180)/2+4)/2+4,((ny-180)/2+4)/2+4,4*chanel_root])
# #     
# #     # Convolution 13
# #     offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
# #     size = tf.to_int32(tf.pack([((nx-186)/2+4)/2+4,((ny-186)/2+4)/2+4]))
# #     h_conv13 = tf.image.extract_glimpse(h_conv6, size=size, offsets=offsets, centered=True)
#     W_conv13 = weight_variable([field_of_view, field_of_view, 8*chanel_root, 4*chanel_root])
#     b_conv13 = bias_variable([4*chanel_root])
#     h_conv13 = tf.nn.relu(conv2d(h_deconv2_concat, W_conv13,keep_prob) + b_conv13)
# #     
#     # Convolution 14
#     W_conv14 = weight_variable([field_of_view, field_of_view, 4*chanel_root, 4*chanel_root])
#     b_conv14 = bias_variable([4*chanel_root])
#     h_conv14 = tf.nn.relu(conv2d(h_conv13, W_conv14,keep_prob) + b_conv14)
#      
#     # Deconvolution 3
#     W_deconv_3 = weight_variable_devonc([max_pool_size, max_pool_size, 2*chanel_root, 4*chanel_root])
#     b_deconv3 = bias_variable([2*chanel_root])
#     h_deconv3 = tf.nn.relu(deconv2d(h_conv14, W_deconv_3, max_pool_size) + b_deconv3)
#     h_deconv3_concat = crop_and_concat(h_conv4,h_deconv3,[batch_size,(nx-180)/2+4,(ny-180)/2+4,2*chanel_root])
 
    # Convolution 15
    offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
    size = tf.to_int32(tf.pack([(nx-184)/2+4,(ny-184)/2+4]))
    h_conv15 = tf.image.extract_glimpse(h_conv4, size=size, offsets=offsets, centered=True)
#     W_conv15 = weight_variable([field_of_view, field_of_view, 4*chanel_root, 2*chanel_root])
#     b_conv15 = bias_variable([2*chanel_root])
#     h_conv15 = tf.nn.relu(conv2d(h_deconv3_concat, W_conv15,keep_prob) + b_conv15)
      
    # Convolution 16
    W_conv16 = weight_variable([field_of_view, field_of_view, 2*chanel_root, 2*chanel_root])
    b_conv16 = bias_variable([2*chanel_root])
    h_conv16 = tf.nn.relu(conv2d(h_conv15, W_conv16,keep_prob) + b_conv16)
      
    # Deconvolution 4
    W_deconv_4 = weight_variable_devonc([max_pool_size, max_pool_size, chanel_root, 2*chanel_root])
    b_deconv4 = bias_variable([chanel_root])
    h_deconv4 = tf.nn.relu(deconv2d(h_conv16, W_deconv_4, max_pool_size) + b_deconv4)
    h_deconv4_concat = crop_and_concat(h_conv2,h_deconv4,[batch_size,nx-180,ny-180,chanel_root])
     
     
    # Convolution 17
#     offsets = tf.zeros(tf.pack([batch_size, 2]), dtype=tf.float32)
#     size = tf.to_int32(tf.pack([nx-182,ny-182]))
#     h_conv17 = tf.image.extract_glimpse(h_conv2, size=size, offsets=offsets, centered=True)
    W_conv17 = weight_variable([field_of_view, field_of_view, 2*chanel_root, chanel_root])
    b_conv17 = bias_variable([chanel_root])
    h_conv17 = tf.nn.relu(conv2d(h_deconv4_concat, W_conv17,keep_prob) + b_conv17)
     
    # Convolution 18
    W_conv18 = weight_variable([field_of_view, field_of_view, chanel_root, chanel_root])
    b_conv18 = bias_variable([chanel_root])
    h_conv18 = tf.nn.relu(conv2d(h_conv17, W_conv18,keep_prob) + b_conv18)
     
    # Output Map
#     W_conv19 = weight_variable(tf.pack([1, 1, chanel_root, n_class]))
    W_conv19 = weight_variable([1, 1, chanel_root, n_class])
    b_conv19 = bias_variable([n_class])
    conv_19 = conv2d(h_conv18, W_conv19,tf.constant(1.0))
    h_conv19 = tf.nn.relu(conv_19 + b_conv19)
    tf.histogram_summary("convolution_19" + '/activations', h_conv19)
     
    tf.image_summary('summary_data', get_image_summary(x_image))
    tf.image_summary('summary_conv_01', get_image_summary(conv_1))
    tf.image_summary('summary_conv_02', get_image_summary(conv_2))
    tf.image_summary('summary_conv_03', get_image_summary(conv_3))
    tf.image_summary('summary_conv_04', get_image_summary(conv_4))
#     tf.image_summary('summary_conv_05', get_image_summary(conv_5))
#     tf.image_summary('summary_conv_06', get_image_summary(conv_6))
#     tf.image_summary('summary_conv_07', get_image_summary(conv_7))
#     tf.image_summary('summary_conv_08', get_image_summary(conv_8))
#     tf.image_summary('summary_conv_09', get_image_summary(conv_9))
#     tf.image_summary('summary_conv_10', get_image_summary(conv_10))
    tf.image_summary('summary_conv_19', get_image_summary(conv_19))
     
    tf.image_summary('summary_pool_01', get_image_summary(h_pool1))
#     tf.image_summary('summary_pool_02', get_image_summary(h_pool2))
#     tf.image_summary('summary_pool_03', get_image_summary(h_pool3))
#     tf.image_summary('summary_pool_04', get_image_summary(h_pool4))
     
#     tf.image_summary('summary_deconv', get_image_summary(deconv_1))
#     tf.image_summary('summary_deconv_concat_01', get_image_summary(h_deconv1_concat))
#     tf.image_summary('summary_deconv_concat_02', get_image_summary(h_deconv2_concat))
#     tf.image_summary('summary_deconv_concat_03', get_image_summary(h_deconv3_concat))
    tf.image_summary('summary_deconv_concat_04', get_image_summary(h_deconv4_concat))
     
     
    return h_conv19
#     output_map = pixel_wise_softmax(h_conv19)
#     return output_map

class Unet(object):
    
    def __init__(self, nx=None, ny=None, channels=3, n_class=2, **kwargs):
        self.n_class = n_class
        
        self.x = tf.placeholder("float", shape=[None, nx, ny, channels])
        self.y = tf.placeholder("float", shape=[None, None, None, n_class])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
#         tf.scalar_summary('dropout_keep_probability',self.keep_prob)
        
        logits = create_conv_net(self.x, self.keep_prob, channels, n_class, **kwargs)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(logits, [-1, n_class]), 
                                                                           tf.reshape(self.y, [-1, n_class])))
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.001  # Choose an appropriate one.
        self.cost = loss + reg_constant * sum(reg_losses)
        
#         tf.scalar_summary('cross entropy', self.cost)
#         self.cost = cross_entropy(self.y, self.net)
#         self.predicter = pixel_wise_softmax(logits)
        self.predicter = pixel_wise_softmax_2(logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.scalar_summary('accuracy', self.accuracy)

    def predict(self, model_path, x_test):
        
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    def save(self, sess, model_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

class Trainer(object):
    
    prediction_path = "prediction"
    
    def __init__(self, net, batch_size=1, momentum=0.9):
        self.net = net
        self.batch_size = batch_size
        self.momentum = momentum
        
    def _initialize(self, training_iters, output_path, restore):
        global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(learning_rate=0.2, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=0.95, 
                                                        staircase=True)
        
        tf.scalar_summary('learning_rate', self.learning_rate)
        tf.scalar_summary('loss', self.net.cost)
        
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
                                                    momentum=self.momentum).minimize(self.net.cost, 
                                                                                     global_step=global_step)
                                                    
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(self.net.cost, 
        #                                            global_step=global_step)
                                                                                     

        self.summary_op = tf.merge_all_summaries()        
        init = tf.initialize_all_variables()
        
        if not restore:
            shutil.rmtree(self.prediction_path)
            shutil.rmtree(output_path)
        
        if not os.path.exists(self.prediction_path):
            os.mkdir(self.prediction_path)
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False):
        
        init = self._initialize(training_iters, output_path, restore)
        save_path = os.path.join(output_path, "model.cpkt")
        
        if epochs == 0:
            return save_path
        
        print("Start optimization")
        
        with tf.Session() as sess:
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            pred_shape = self.store_prediction(sess, data_provider, "start")
            
            summary_writer = tf.train.SummaryWriter(output_path, graph=sess.graph)
            
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                     
                    # Run optimization op (backprop)
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate), feed_dict={self.net.x: batch_x,  
                                                                    self.net.y: crop_to_shape(batch_y, pred_shape),
                                                                    self.net.keep_prob: dropout})
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, crop_to_shape(batch_y, pred_shape))
                        
                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, data_provider, epoch)
                    
                save_path = self.net.save(sess, save_path)
            print("Optimization Finished!")
            
            return save_path
        
    def store_prediction(self, sess, data_provider, epoch):
        batch_x, batch_y = data_provider(4)
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1.})
        print("Epoch error= {:.1f}%".format(error_rate(prediction, batch_y)))
              
        img = combine_img_prediction(batch_x, batch_y, prediction)
        Image.fromarray(img).save("%s/epoch_%s.png"%(self.prediction_path, epoch))
        return prediction.shape
    
    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        print("Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))
    
    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op, 
                                                            self.net.cost, 
                                                            self.net.accuracy, 
                                                            self.net.predicter], 
                                                           feed_dict={self.net.x: batch_x,
                                                                      self.net.y: batch_y,
                                                                      self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        print("Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(step,
                                                                                                            loss,
                                                                                                            acc,
                                                                                                            error_rate(predictions, batch_y)))


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
        (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))


def get_image_summary(img, idx=0):
    """Make an image summary for 4d tensor image with index idx"""
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.pack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.pack((-1, img_w, img_h, 1)))
    return V
