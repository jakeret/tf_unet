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
Created on Aug 18, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

from PIL import Image
import numpy as np


class DataProvider(object):
    channels = 1
    n_class = 2
    
    def __init__(self,  data_files, mask_files, a_min=0, a_max=210):
        self.data_files = data_files
        self.mask_files = mask_files
        self.a_min = a_min
        self.a_max = a_max
        
        assert len(data_files) > 0, "No training data_files"
        assert len(data_files) == len(mask_files), "data!=mask"
        print("Number of data_files used: %s"%len(data_files))
    
    def _read_file(self):
        data = np.array(Image.open(self.data_files[self.file_idx]), dtype=np.float32)
        mask = np.array(Image.open(self.mask_files[self.file_idx]), dtype=np.bool)
        return data, mask

    def _next_chunck(self):
        self._cylce_file()
        data, mask = self._read_file()
        while mask.sum() == 0:
            self._cylce_file()
            data, mask = self._read_file()
            
        return data, mask

    
    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.data_files))
        

    def _load_data_and_label(self):
        data, mask = self._next_chunck()
        nx = data.shape[1]
        ny = data.shape[0]
        
        #normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        train_data = data.reshape(1, ny, nx, 1)
        train_data -= np.amin(data)
        train_data /= np.amax(data)
        labels = np.zeros((1, ny, nx, 2), dtype=np.float32)
        labels[..., 1] = mask
        labels[..., 0] = ~mask
        
        return train_data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n,nx,ny, 1))
        Y = np.zeros((n,nx,ny,2))

        X[0] = train_data
        Y[0] = labels
        
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
        return X, Y

