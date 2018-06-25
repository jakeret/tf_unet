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
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util


if __name__ == '__main__':
    np.random.seed(98765)

    generator = image_gen.GrayScaleDataProvider(nx=572, ny=572, cnt=20, rectangles=False)
    
    net = unet.Unet(channels=generator.channels, 
                    n_class=generator.n_class, 
                    layers=3,
                    features_root=16)
    
    trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
    path = trainer.train(generator, "./unet_trained",
                         training_iters=32,
                         epochs=5,
                         dropout=0.75,# probability to keep units
                         display_step=2)
     
    x_test, y_test = generator(4)
    prediction = net.predict(path, x_test)
     
    print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction,
                                                               util.crop_to_shape(y_test, prediction.shape))))
