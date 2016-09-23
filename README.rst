=============================
Tensorflow Unet
=============================

.. image:: https://readthedocs.org/projects/tf-unet/badge/?version=latest
	:target: http://tf-unet.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status
		

This is a generic convolutional neural network implementation following the **U-Net** architecture proposed in this paper (https://arxiv.org/pdf/1505.04597.pdf).

The code is not tied to a specific segmentation such that it has been used in a toy problem to detect circles in a noisy image.

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


Using `tf_unet` is easy! Checkout the *Usage* section or the included Jupyter Notebook.