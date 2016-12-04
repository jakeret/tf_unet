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

import h5py
import numpy as np
from tf_unet.image_util import BaseDataProvider


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
        
