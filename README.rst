=============================
Tensorflow Unet
=============================

.. image:: https://readthedocs.org/projects/tf-unet/badge/?version=latest
	:target: http://tf-unet.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status
		
.. image:: http://img.shields.io/badge/arXiv-1609.09077-orange.svg?style=flat
        :target: http://arxiv.org/abs/1609.09077

.. image:: https://img.shields.io/badge/ascl-1611.002-blue.svg?colorB=262255
        :target: http://ascl.net/1611.002


This is a generic convolutional neural network implementation following the **U-Net** architecture proposed in this `paper <https://arxiv.org/pdf/1505.04597.pdf>`_ written with **Tensorflow**. The code has been developed and used for `Radio Frequency Interference mitigation using deep convolutional neural networks <http://arxiv.org/abs/1609.09077>`_ 

Checkout the *Usage* section or the included Jupyter notebooks for a `toy problem <https://github.com/jakeret/tf_unet/blob/master/demo/demo_toy_problem.ipynb>`_ or the `RFI problem <https://github.com/jakeret/tf_unet/blob/master/demo/demo_radio_data.ipynb>`_ discussed in the paper.

The code is not tied to a specific segmentation such that it can be used in a toy problem to detect circles in a noisy image.

.. image:: https://raw.githubusercontent.com/jakeret/tf_unet/master/docs/toy_problem.png
   :alt: Segmentation of a toy problem.
   :align: center

To more complex application such as the detection of radio frequency interference (RFI) in radio astronomy.

.. image:: https://raw.githubusercontent.com/jakeret/tf_unet/master/docs/rfi.png
   :alt: Segmentation of RFI in radio data.
   :align: center

Or to detect galaxies and star in wide field imaging data.

.. image:: https://raw.githubusercontent.com/jakeret/tf_unet/master/docs/galaxies.png
   :alt: Segmentation of a galaxies.
   :align: center


