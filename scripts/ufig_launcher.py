# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import os
import click

from tf_unet import unet
from tf_unet import util

from scripts.ufig_util import DataProvider

def create_training_path(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path

@click.command()
@click.option('--data_root', default="./ufig_images/1.h5")
@click.option('--output_path', default="./unet_trained_ufig")
@click.option('--training_iters', default=20)
@click.option('--epochs', default=10)
@click.option('--restore', default=False)
@click.option('--layers', default=3)
@click.option('--features_root', default=16)
def launch(data_root, output_path, training_iters, epochs, restore, layers, features_root):
    data_provider = DataProvider(572, data_root)
    
    data, label = data_provider(1)
    weights = None#(1/3) / (label.sum(axis=2).sum(axis=1).sum(axis=0) / data.size)
    
    net = unet.Unet(channels=data_provider.channels, 
                    n_class=data_provider.n_class, 
                    layers=layers, 
                    features_root=features_root,
                    cost_kwargs=dict(regularizer=0.001,
                                     class_weights=weights),
                    )
    
    path = output_path if restore else create_training_path(output_path)
    trainer = unet.Trainer(net, optimizer="adam", opt_kwargs=dict(beta1=0.91))
    path = trainer.train(data_provider, path, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
                         display_step=2, 
                         restore=restore)
     
    prediction = net.predict(path, data)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))))
    

if __name__ == '__main__':
    launch()