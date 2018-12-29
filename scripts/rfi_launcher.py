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

Trains a tf_unet network to segment radio frequency interference pattern.
Requires data from the Bleien Observatory or a HIDE&SEEK simulation.
'''

from __future__ import print_function, division, absolute_import, unicode_literals
import glob
import click
import h5py
import numpy as np

from tf_unet import unet
from tf_unet import util
from tf_unet.image_util import BaseDataProvider


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
    data_provider = DataProvider(600, glob.glob(data_root+"/*"))
    
    net = unet.Unet(channels=data_provider.channels, 
                    n_class=data_provider.n_class, 
                    layers=layers, 
                    features_root=features_root,
                    cost_kwargs=dict(regularizer=0.001),
                    )
    
    path = output_path if restore else util.create_training_path(output_path)
    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(data_provider, path, 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=0.5, 
                         display_step=2, 
                         restore=restore)
     
    x_test, y_test = data_provider(1)
    prediction = net.predict(path, x_test)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
    

class DataProvider(BaseDataProvider):
    """
    Extends the BaseDataProvider to randomly select the next
    data chunk
    """

    channels = 1
    n_class = 2

    def __init__(self, nx, files, a_min=30, a_max=210):
        super(DataProvider, self).__init__(a_min, a_max)
        self.nx = nx
        self.files = files

        assert len(files) > 0, "No training files"
        print("Number of files used: %s"%len(files))
        self._cylce_file()

    def _read_chunck(self):
        with h5py.File(self.files[self.file_idx], "r") as fp:
            nx = fp["data"].shape[1]
            idx = np.random.randint(0, nx - self.nx)

            sl = slice(idx, (idx+self.nx))
            data = fp["data"][:, sl]
            rfi = fp["mask"][:, sl]
        return data, rfi

    def _next_data(self):
        data, rfi = self._read_chunck()
        nx = data.shape[1]
        while nx < self.nx:
            self._cylce_file()
            data, rfi = self._read_chunck()
            nx = data.shape[1]

        return data, rfi

    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.files))


if __name__ == '__main__':
    launch()
