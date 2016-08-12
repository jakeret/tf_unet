# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util

if __name__ == '__main__':
    nx = 572
    ny = 572
    channels = 1
    n_class = 2
     
    n_image = 1 #batch size
    training_iters = 20
    epochs = 10
    dropout = 0.75 # Dropout, probability to keep units
    display_step = 2
    restore = False
 
    generator = image_gen.get_image_gen(nx, ny)
    
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
