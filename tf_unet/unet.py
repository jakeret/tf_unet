# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import os
import shutil
import numpy as np
from tf_unet import util

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
#     initial = 1.0/float(shape[0]*shape[1])
    #return tf.Variable(tf.add((np.ones(shape)*initial).astype(float32),tf.truncated_normal(shape, stddev=0.1)))
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

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

def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
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
    
    stddev = np.sqrt(2 / (filter_size**2 * features_root))
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
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
            pools[layer] = max_pool(h_convs[layer], pool_size)
            in_node = pools[layer]
        
    in_node = h_convs[layers-1]
        
    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(h_convs[layer], h_deconv, [batch_size])
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
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
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    output_map = tf.nn.relu(conv + bias)
    
    if summaries:
        for i, (c1, c2) in enumerate(convs):
            tf.image_summary('summary_conv_%02d_01'%i, get_image_summary(c1))
            tf.image_summary('summary_conv_%02d_02'%i, get_image_summary(c2))
            
        for k in sorted(pools.keys()):
            tf.image_summary('summary_pool_%02d'%k, get_image_summary(pools[k]))
        
        for k in sorted(deconv.keys()):
            tf.image_summary('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))
        
    return output_map


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
            shutil.rmtree(self.prediction_path, ignore_errors=True)
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(self.prediction_path):
            os.mkdir(self.prediction_path)
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.75, display_step=1, restore=False):
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init = self._initialize(training_iters, output_path, restore)
        
        print("Start optimization")
        
        with tf.Session() as sess:
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            test_x, test_y = data_provider(4)
            pred_shape = self.store_prediction(sess, test_x, test_y, "start")
            
            summary_writer = tf.train.SummaryWriter(output_path, graph=sess.graph)
            
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)
                     
                    # Run optimization op (backprop)
                    _, loss, lr = sess.run((self.optimizer, self.net.cost, self.learning_rate), feed_dict={self.net.x: batch_x,  
                                                                    self.net.y: util.crop_to_shape(batch_y, pred_shape),
                                                                    self.net.keep_prob: dropout})
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, util.crop_to_shape(batch_y, pred_shape))
                        
                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, epoch)
                    
                save_path = self.net.save(sess, save_path)
            print("Optimization Finished!")
            
            return save_path
        
    def store_prediction(self, sess, batch_x, batch_y, epoch):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x, self.net.y: batch_y, self.net.keep_prob: 1.})
        print("Prediction error= {:.1f}%".format(error_rate(prediction, 
                                                       util.crop_to_shape(batch_y, 
                                                                          prediction.shape))))
              
        img = util.combine_img_prediction(batch_x, batch_y, prediction)
        util.save_image(img, "%s/epoch_%s.jpg"%(self.prediction_path, epoch))
        
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
