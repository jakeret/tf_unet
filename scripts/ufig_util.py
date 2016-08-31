# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Aug 30, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from astropy.io import fits

class Generator(object):
    channels = 1
    n_class = 3
    
    def __init__(self, nx, root_path, a_min=0, a_max=20):
        self.nx = nx
        self.root_path = root_path
        self.a_min = a_min
        self.a_max = a_max
        
        self._load_data()
        
    def _load_data(self):
        self.image = fits.getdata(self.root_path + ".fits")
        self.gal_map = fits.getdata(self.root_path + ".catalog_galaxy_seg.fits")
        self.star_map = fits.getdata(self.root_path + ".catalog_star_seg.fits")

    def _transpose_3d(self, a):
        return np.stack([a[..., i].T for i in range(a.shape[2])], axis=2)
        
    def _transform(self, data, labels):
        op = np.random.randint(0, 4)
        if op == 0:
            if np.random.randint(0, 2) == 0:
                data, labels = self._transpose_3d(data), self._transpose_3d(labels)
        else:    
            data, labels = np.rot90(data, op), np.rot90(labels, op)
            
        return data, labels
            
            
    
    def _load_data_and_label(self):
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
        
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return self._transform(data[:,:, np.newaxis], labels)

    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[0]
        ny = train_data.shape[1]

        X = np.zeros((n,nx,ny, self.channels))
        Y = np.zeros((n,nx,ny, self.n_class))
        
        X[0] = train_data
        Y[0] = labels
        
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels
        return X, Y
