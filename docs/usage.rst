========
Usage
========

To use Tensorflow Unet in a project::

	from tf_unet import unet, util, image_util
	
	#preparing data loading
	data_provider = image_util.ImageDataProvider("fishes/train/*.tif")

	#setup & training
	net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
	trainer = unet.Trainer(net)
	path = trainer.train(data_provider, output_path, training_iters=32, epochs=100)
	
	#verification
	...
	
	prediction = net.predict(path, data)
	
	unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
	
	img = util.combine_img_prediction(data, label, prediction)
	util.save_image(img, "prediction.jpg")
	
Keep track of the learning progress using *Tensorboard*. **tf_unet** automatically outputs relevant summaries.

.. image:: https://raw.githubusercontent.com/jakeret/tf_unet/master/docs/stats.png
   :alt: Segmentation of a toy problem.
   :align: center


More examples can be found in the Jupyter notebooks for a `toy problem <https://github.com/jakeret/tf_unet/blob/master/demo/demo_toy_problem.ipynb>`_ or for a `RFI problem <https://github.com/jakeret/tf_unet/blob/master/demo/demo_radio_data.ipynb>`_.
Further code is stored in the `scripts <https://github.com/jakeret/tf_unet/tree/master/scripts>`_ folder.
