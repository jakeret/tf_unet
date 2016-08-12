# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Aug 10, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_prediction(x_test, y_test, prediction, save=False):
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)
    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i, 92:-92, 92:-92])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()
    
    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()

def to_rgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
        
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def combine_img_prediction(data, gt, pred):
    img = np.concatenate((to_rgb(data[0, 92:-92, 92:-92]), 
                          to_rgb(gt[0, ..., 1]), 
                          to_rgb(pred[0, ..., 1])), axis=1)
    return img.round().astype(np.uint8)
