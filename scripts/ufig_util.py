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
Created on Aug 30, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from tf_unet.image_util import BaseDataProvider

class DataProvider(BaseDataProvider):
    """
    Extends the BaseDataProvider to randomly select the next 
    chunk of the image and randomly applies transformations to the data
    """

    channels = 1
    n_class = 3
    
    def __init__(self, nx, path, a_min=0, a_max=20, sigma=1):
        super(DataProvider, self).__init__(a_min, a_max)
        self.nx = nx
        self.path = path
        self.sigma = sigma
        
        self._load_data()
        
    def _load_data(self):
        with h5py.File(self.path, "r") as fp:
            self.image = gaussian_filter(fp["image"].value, self.sigma)
            self.gal_map = fp["segmaps/galaxy"].value
            self.star_map = fp["segmaps/star"].value

    def _transpose_3d(self, a):
        return np.stack([a[..., i].T for i in range(a.shape[2])], axis=2)
        
    def _post_process(self, data, labels):
        op = np.random.randint(0, 4)
        if op == 0:
            if np.random.randint(0, 2) == 0:
                data, labels = self._transpose_3d(data[:,:,np.newaxis]), self._transpose_3d(labels)
        else:    
            data, labels = np.rot90(data, op), np.rot90(labels, op)
            
        return data, labels
    
    def _next_data(self):
        ix = np.random.randint(0, self.image.shape[0] - self.nx)
        iy = np.random.randint(0, self.image.shape[1] - self.nx)
        
        slx = slice(ix, ix+self.nx)
        sly = slice(iy, iy+self.nx)
        
        data = self.image[slx, sly]
        gal_seg = self.gal_map[slx, sly]
        star_seg = self.star_map[slx, sly]
        
        labels = np.zeros((self.nx, self.nx, self.n_class), dtype=np.float32)
        labels[..., 1] = np.clip(gal_seg, 0, 1)
        labels[..., 2] = np.clip(star_seg, 0, 1)
        labels[..., 0] = (1+np.clip(labels[...,1] + labels[...,2], 0, 1))%2
        
        return data, labels
    
