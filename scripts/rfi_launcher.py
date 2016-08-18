# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet import unet
from tf_unet import util

import glob
from scripts.radio_util import Generator

DATA_ROOT = ".cache/"

if __name__ == '__main__':
    training_iters = 20
    epochs = 100
    dropout = 0.75 # Dropout, probability to keep units
    display_step = 2
    restore = False
 
    generator = Generator(600, glob.glob(DATA_ROOT+"*"))
    
    net = unet.Unet(channels=generator.channels, 
                    n_class=generator.n_class, 
                    layers=3, 
                    features_root=16)
    
    trainer = unet.Trainer(net, momentum=0.2)
    path = trainer.train(generator, "../unet_trained", 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=dropout, 
                         display_step=display_step, 
                         restore=restore)
     
    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
    
#     import numpy as np
#     np.savetxt("prediction.txt", prediction[..., 1].reshape(-1, prediction.shape[2]))
    
    img = util.combine_img_prediction(x_test, y_test, prediction)
    util.save_image(img, "prediction.jpg")
