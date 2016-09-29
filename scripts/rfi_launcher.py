# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import glob
import click

from tf_unet import unet
from tf_unet import util

from scripts.radio_util import Generator

def create_training_path(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path

@click.command()
@click.option('--data_root', default="./bleien_data")
@click.option('--output_path', default="./daint_unet_trained_rfi_bleien")
@click.option('--training_iters', default=32)
@click.option('--epochs', default=100)
@click.option('--restore', default=False)
@click.option('--layers', default=5)
@click.option('--features_root', default=64)
def launch(data_root, output_path, training_iters, epochs, restore, layers, features_root):
    print("Using data from: %s"%data_root)
    generator = Generator(600, glob.glob(data_root+"/*"))
    
    net = unet.Unet(channels=generator.channels, 
                    n_class=generator.n_class, 
                    layers=layers, 
                    features_root=features_root,
                    add_regularizers=True,
#                     filter_size=5
                    )
    
    path = output_path if restore else create_training_path(output_path)
    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(generator, path, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
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