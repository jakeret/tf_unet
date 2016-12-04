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

import numpy as np
from tf_unet.image_util import ImageDataProvider

class DataProvider(ImageDataProvider):
    """
    Extends the default ImageDataProvider to randomly select the next 
    image and ensures that only data sets are used where the mask is not empty
    """

    def _next_data(self):
        data, mask = super(DataProvider, self)._next_data()
        while mask.sum() == 0:
            self._cylce_file()
            data, mask = super(DataProvider, self)._next_data()
            
        return data, mask

    
    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.data_files))
        
