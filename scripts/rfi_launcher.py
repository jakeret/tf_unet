# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from tf_unet import unet
from tf_unet import util

import glob
import h5py
import numpy as np

DATA_ROOT = ".cache/"
class Generator(object):
    
    def __init__(self, nx, a_min=30, a_max=210):
        self.nx = nx
        self.a_min = a_min
        self.a_max = a_max
        self.files = glob.glob(DATA_ROOT+"*")
        self.file_idx = 0
        self.iter = 0
    
    def _read_chunck(self):
        with h5py.File(self.files[self.file_idx], "r") as fp:
            sl = slice((self.iter*self.nx), ((self.iter+1)*self.nx))
            data = fp["data"][:, sl]
            rfi = fp["RFI"][:, sl]
        return data, rfi
    
    def _next_chunck(self):
        data, rfi = self._read_chunck()
        self.iter += 1
        nx = data.shape[1]
        while nx < self.nx:
            self._cylce_file()
            data, rfi = self._read_chunck()
            nx = data.shape[1]
            self.iter += 1
            
        return data, rfi

    def _cylce_file(self):
        self.iter = 0
        self.file_idx += 1
        if self.file_idx >= len(self.files):
            self.file_idx = 0

    def __call__(self, n):
        data, rfi = self._next_chunck()
        nx = data.shape[1]
        ny = data.shape[0]
        data = np.clip(data, self.a_min, self.a_max)
        train_data = data.reshape(1, ny, nx, 1)
        train_data -= np.amin(data)
        train_data /= np.amax(data)
        labels = np.zeros((1, ny, nx, 2), dtype=np.float32)
        labels[..., 1] = rfi
        labels[..., 0] = ~rfi
        return train_data, labels
        
    


if __name__ == '__main__':
    channels = 1
    n_class = 2
     
    training_iters = 20
    epochs = 100
    dropout = 0.75 # Dropout, probability to keep units
    display_step = 2
    restore = False
 
    generator = Generator(600)
    
    net = unet.Unet(channels=channels, n_class=n_class, layers=3, features_root=16)
    
    trainer = unet.Trainer(net, momentum=0.2)
    path = trainer.train(generator, "./unet_trained", 
                         training_iters=training_iters, 
                         epochs=epochs, 
                         dropout=dropout, 
                         display_step=display_step, 
                         restore=restore)
     
    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))
    
    import numpy as np
    np.savetxt("prediction.txt", prediction[..., 1].reshape(-1, prediction.shape[2]))
    
    img = util.combine_img_prediction(x_test, y_test, prediction)
    util.save_image(img, "prediction.jpg")
