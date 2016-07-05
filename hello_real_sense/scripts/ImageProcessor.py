#!/usr/bin/env python

# load system packages
import time
import os
import numpy as np
import cv2
import sys

# load ros packages
import rospy
from std_msgs.msg import String
import roslib
roslib.load_manifest('hello_real_sense')
import rospkg

# load other ROS packages for computer vision
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# load your machine learning packages
import keras.models as models
from keras.layers.core import Layer, Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

class ImageProcessor:
	'''
		A simple Image Processing class to process your image using a Conv Denconv Variational Autoencoder
		Research Paper idea: http://mi.eng.cam.ac.uk/projects/segnet/#publication
		Deep Learning model idea is borrowed from https://github.com/pradyu1993/segnet
	'''
	def __init__(self):
		data_shape = 360*480
		curRosPackage = rospkg.RosPack()
		self.curModulePath = curRosPackage.get_path('hello_real_sense')
		print 'creating the classifier model'
		# initialize
		autoEncoderModel = models.Sequential()
		# create the input layer
		autoEncoderModel.add(Layer(input_shape=(3, 360, 480)))
		# create the encoding layer
		autoEncoderModel.encoding_layers = self.create_encoding_layers()
		# create the decoding layer
		autoEncoderModel.decoding_layers = self.create_decoding_layers()

		# add the encoding layers to 
		for l in autoEncoderModel.encoding_layers:
			autoEncoderModel.add(l)
		# then add decoding layers
		for l in autoEncoderModel.decoding_layers:
			autoEncoderModel.add(l)

		# set colors for the mask based on the classifier output
		Unlabelled = [0,0,0]
		Pavement = [255,255,255]


		self.label_colours = np.array([Unlabelled, Unlabelled, Unlabelled, Pavement,
			Unlabelled, Unlabelled, Unlabelled, Unlabelled, Unlabelled, Unlabelled, 
			Unlabelled, Unlabelled])

		autoEncoderModel.add(Convolution2D(12, 1, 1, border_mode='valid',))
		autoEncoderModel.add(Reshape((12,data_shape), input_shape=(12,360,480)))
		autoEncoderModel.add(Permute((2, 1)))
		autoEncoderModel.add(Activation('softmax'))

		print 'loading weights for classifier'
		# autoEncoderModel.save_weights('model_weight_ep100.hdf5')
		autoEncoderModel.load_weights(self.curModulePath+'/classifier/seg_net_weights.hdf5')
		self.imgClassifier = autoEncoderModel
		print 'done loading weights'

	def processMyImage(self, imgToProcess):
		'''
			Function to process the incoming message
				take the image
				split the image into different channels
				initialize tensor
				pass to classifier
				retreive output and set create a mask
				overlap mask with original image and return
		'''
		# load the image
		img = cv2.resize(imgToProcess, (480, 360))
		# split rgb
		imgB, imgG, imgR = cv2.split(img)
		newImg = np.zeros((1, 3, 360, 480))
		# put them in 4D tensor
		newImg[0, 0,:,:] = imgB
		newImg[0, 1,:,:] = imgG
		newImg[0, 2,:,:] = imgR

		print 'predicing image'
		time_start = time.clock()
		preval = self.imgClassifier.predict(newImg)
		print 'time taken: ', time.clock() - time_start, 'secs'
		
		# visualize the segmented image with colors
		newImg = self.visualizeSegmentedImage(np.argmax(preval[0],axis=1).reshape((360,480)))
		# resize image back
		newImg = cv2.resize(newImg, (640, 480))

		# consider to process certain portion of the image
		# ignore top portion of the image
		newImg[0:200,:,:] = 0
		# cv2.imshow("before erosion and dilation", newImg.copy())
		kernel = np.ones((9,9),np.uint8)
		newImg = cv2.erode(newImg,kernel,iterations = 1)
		newImg = cv2.dilate(newImg,kernel,iterations = 1)
		
		correctedImg = self.overlapMask(imgToProcess, newImg)
		correctedImg = correctedImg.astype(np.uint8)
		# cv2.imshow("correctedImg", correctedImg)
		# cv2.imshow("eroded dilated", newImg)
		# cv2.waitKey(4)
		return correctedImg

	def overlapMask(self, origImg, processedImg):
		'''
			Take original image, take the mask and overlap them
		'''
		origB, origG, origR = cv2.split(origImg)
		processedImg[processedImg != 1] = 0
		processedB, processedG, processedR = cv2.split(processedImg)

		# mark red BGR
		origB[processedB == 1] = 0
		origG[processedG == 1] = 0
		origR[processedR == 1] = 255

		# reset original values
		origImg[:,:,0] = origB
		origImg[:,:,1] = origG
		origImg[:,:,2] = origR

		return origImg

	def visualizeSegmentedImage(self, maskedImg):
		'''
			based on classifier output, create the mask
		'''
		b = maskedImg.copy()
		g = maskedImg.copy()
		r = maskedImg.copy()
		for l in range(0,11):
			b[maskedImg==l]=self.label_colours[l,0]
			g[maskedImg==l]=self.label_colours[l,1]
			r[maskedImg==l]=self.label_colours[l,2]
		
		bgr = np.zeros((maskedImg.shape[0], maskedImg.shape[1], 3))
		bgr[:,:,0] = (b/255.0)#[:,:,0]
		bgr[:,:,1] = (g/255.0)#[:,:,1]
		bgr[:,:,2] = (r/255.0)#[:,:,2]
		bgr[np.where(bgr < 1)] = 0.0

		return bgr

	def create_encoding_layers(self):
		'''
			function to return the set of encoding layers for the deep learning model
		'''
		kernel = 3
		filter_size = 64
		pad = 1
		pool_size = 2
		return [
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(pool_size, pool_size)),

			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(128, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(pool_size, pool_size)),

			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(256, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(pool_size, pool_size)),

			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(512, kernel, kernel, border_mode='valid'),
			BatchNormalization(),
			Activation('relu')
		]

	def create_decoding_layers(self):
		'''
			function to return the set of decoders layers for the deep learning model
		'''
		kernel = 3
		filter_size = 64
		pad = 1
		pool_size = 2
		return[
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(512, kernel, kernel, border_mode='valid'),
			BatchNormalization(),

			UpSampling2D(size=(pool_size,pool_size)),
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(256, kernel, kernel, border_mode='valid'),
			BatchNormalization(),

			UpSampling2D(size=(pool_size,pool_size)),
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(128, kernel, kernel, border_mode='valid'),
			BatchNormalization(),

			UpSampling2D(size=(pool_size,pool_size)),
			ZeroPadding2D(padding=(pad,pad)),
			Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
			BatchNormalization()
		]
	pass