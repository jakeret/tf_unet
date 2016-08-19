# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Aug 18, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import h5py
import numpy as np


class Generator(object):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, files, a_min=30, a_max=210):
        self.nx = nx
        self.files = files
        self.a_min = a_min
        self.a_max = a_max
        
        assert len(files) > 0, "No training files"
        print("Number of files used: %s"%len(files))
        self._cylce_file()
    
    def _read_chunck(self):
        with h5py.File(self.files[self.file_idx], "r") as fp:
            nx = fp["data"].shape[1]
            idx = np.random.choice(np.arange(nx/self.nx).astype(np.int))
            
            sl = slice((idx*self.nx), ((idx+1)*self.nx))
            data = fp["data"][:, sl]
            rfi = fp["mask"][:, sl]
        return data, rfi
    
    def _next_chunck(self):
        data, rfi = self._read_chunck()
        nx = data.shape[1]
        while nx < self.nx:
            self._cylce_file()
            data, rfi = self._read_chunck()
            nx = data.shape[1]
            
        return data, rfi

    def _cylce_file(self):
        self.file_idx = np.random.choice(len(self.files))
        

    def _load_data_and_label(self):
        data, rfi = self._next_chunck()
        nx = data.shape[1]
        ny = data.shape[0]
        
        #normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        train_data = data.reshape(1, ny, nx, 1)
        train_data -= np.amin(data)
        train_data /= np.amax(data)
        labels = np.zeros((1, ny, nx, 2), dtype=np.float32)
        labels[..., 1] = rfi
        labels[..., 0] = ~rfi
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

