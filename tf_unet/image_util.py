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
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
from PIL import Image

class ImageDataProvider(object):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) clip min
    :param a_max: (optional) clip max
    :param mask_suffix: suffix pattern for the label images
    
    """
    n_class = 2
    
    def __init__(self, search_path, a_min=-np.inf, a_max=np.inf, mask_suffix='_mask.tif'):
        self.a_min = a_min
        self.a_max = a_max
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        
        self.data_files = self.find_data_files(search_path)
    
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))
        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if not self.mask_suffix in name]
    
    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(".tif", self.mask_suffix)
        
        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)
    
        return img,label
    
    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx > len(self.data_files):
            self.file_idx = 0 
        

    
    def _load_data_and_label(self):
        data, label = self._next_data()
            
        nx = data.shape[1]
        ny = data.shape[0]
    
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        train_data = data.reshape(1, ny, nx, self.channels)
        train_data -= np.amin(data)
        train_data /= np.amax(data)
        labels = np.zeros((1,ny, nx, self.n_class), dtype=np.float32)
        labels[..., 1] = label
        labels[..., 0] = ~label
        
        return self.post_process(train_data, labels)
    
    def post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]
    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))
    
        X[0] = train_data
        Y[0] = labels
        for i in range(0, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
    
        return X, Y
    
    
