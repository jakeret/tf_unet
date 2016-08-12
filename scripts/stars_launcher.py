# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import my_library_stars as my_lib
from tf_unet import unet
from tf_unet import util


STREET_DATA_ROOT = '/Users/jakeret/workspace/CNN_stars/data/'
NUM_BIG_IMAGES = 2

def create_star_generator():
    all_train_data = my_lib.data_s_resh(STREET_DATA_ROOT, 1, NUM_BIG_IMAGES, resize_factor = 4)
    train_data = all_train_data.img_array  # should change this later: we don't need to save it twice......
    train_labels = all_train_data.gt_1hot_array
    
    img_cnt = train_data.shape[0]
    def generator(n):
        idx = np.random.randint(0, img_cnt)
        return train_data[[idx]], train_labels[[idx], 92:-92, 92:-92,:]
    
    return generator

if __name__ == '__main__':
    n_class = 2
     
    n_image = 1 #batch size
    training_iters = 20
    epochs = 100
    dropout = 0.75 # Dropout, probability to keep units
    display_step = 2
    restore = False
 
    generator = create_star_generator()
    x_test, y_test = generator(1)
    nx = x_test.shape[1]
    ny = x_test.shape[2]
    channels = x_test.shape[3]

    net = unet.Unet(nx, ny, channels, n_class)
    
    trainer = unet.Trainer(net, batch_size=n_image)
    path = trainer.train(generator, "./unet_trained", 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=dropout, 
                         display_step=display_step, 
                         restore=restore)
     
    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, y_test)))
    util.plot_prediction(x_test[..., 0], y_test, prediction)

