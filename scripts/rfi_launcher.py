# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import glob
import click

from tf_unet import unet
from tf_unet import util

from scripts.radio_util import Generator

@click.command()
@click.option('--data_root', default="./bleien_data")
@click.option('--output_path', default="./daint_unet_trained_rfi_bleien")
@click.option('--training_iters', default=32)
@click.option('--epochs', default=100)
@click.option('--restore', default=False)
@click.option('--layers', default=3)
@click.option('--features_root', default=16)
def launch(data_root, output_path, training_iters, epochs, restore, layers, features_root):
    print("Using data from: %s"%data_root)
    generator = Generator(2400, glob.glob(data_root+"/*"))
    
    net = unet.Unet(channels=generator.channels, 
                    n_class=generator.n_class, 
                    layers=layers, 
                    features_root=features_root
                    )
    
    trainer = unet.Trainer(net, momentum=0.2)
    path = trainer.train(generator, output_path, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.75, 
                         display_step=2, 
                         restore=restore)
     
    x_test, y_test = generator(1)
    prediction = net.predict(path, x_test)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
    
#     import numpy as np
#     np.save("prediction", prediction[0, ..., 1])
    
    img = util.combine_img_prediction(x_test, y_test, prediction)
    util.save_image(img, "prediction.jpg")


if __name__ == '__main__':
    launch()