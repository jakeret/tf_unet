# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np

sigma = 10

plateau_min = -2
plateau_max = 2

r_min = 1
r_max = 200

def create_image_and_label(nx,ny, cnt = 10):
    r_min = 5
    r_max = 50
    border = 92
    sigma = 20
    
    image = np.ones((ny, nx, 1))
    label = np.ones((ny, nx))
    mask = np.zeros((nx,ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y,x = np.ogrid[-a:ny-a, -b:nx-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h
    label[mask] = 0
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    
    return image, label

def get_image_gen(nx, ny):
    def create_batch(n_image):
        
        X = np.zeros((n_image,nx,ny, 1))
        Y = np.zeros((n_image,nx,ny,2))
        
        for i in range(n_image):
            X[i],Y[i,:,:,1] = create_image_and_label(nx,ny)
            Y[i,:,:,0] = 1-Y[i,:,:,1]
            
        return X,Y
    return create_batch

def get_image_gen_rgb(nx, ny, **kwargs):
    def create_batch(n_image):
            
            X = np.zeros((n_image, nx, ny, 3))
            Y = np.zeros((n_image, nx, ny,2))
            
            for i in range(n_image):
                x, Y[i,:,:,1] = create_image_and_label(nx,ny, **kwargs)
                X[i] = to_rgb(x)
                Y[i,:,:,0] = 1-Y[i,:,:,1]
                
            return X, Y
    return create_batch

def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

# def create_image_and_label(nx,ny):
#     x = np.floor(np.random.rand(1)[0]*nx).astype(np.int)
#     y = np.floor(np.random.rand(1)[0]*ny).astype(np.int)
# 
#     image = np.ones((nx,ny))
#     label = np.ones((nx,ny))
#     image[x,y] = 0
#     image_distance = ndimage.morphology.distance_transform_edt(image)
# 
#     r = np.random.rand(1)[0]*(r_max-r_min)+r_min
#     plateau = np.random.rand(1)[0]*(plateau_max-plateau_min)+plateau_min
# 
#     label[image_distance <= r] = 0 
#     label[image_distance > r] = 1
#     label = (1 - label)
#     
#     image_distance[image_distance <= r] = 0 
#     image_distance[image_distance > r] = 1
#     image_distance = (1 - image_distance)*plateau
# 
#     image = image_distance + np.random.randn(nx,ny)/sigma
#     
#     return image, label[92:nx-92,92:nx-92]
# 
# def get_image_gen(nx, ny):
#     def create_batch(n_image):
#         
#         X = np.zeros((n_image,nx,ny))
#         Y = np.zeros((n_image,nx-184,ny-184,2))
#         
#         for i in range(n_image):
#             X[i,:,:],Y[i,:,:,1] = create_image_and_label(nx,ny)
#             Y[i,:,:,0] = 1-Y[i,:,:,1]
#             
#         return X,Y
#     return create_batch