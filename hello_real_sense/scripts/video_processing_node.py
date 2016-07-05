#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv2

import roslib
roslib.load_manifest('hello_real_sense')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from ImageProcessor import ImageProcessor
import rospkg

class RosImgToOpenCVImageConvertor:
	'''
		This class does:
			OPENCV ROS Image bridge by initializing sidewalk_detector node
			Initializes Deep Learning classifier
			subscribes image from  topic /camera/color/image_raw
			publishes Image data onto /sidewalk_detector/color/image_raw TOPIC
	'''
	def __init__(self):

		
		# initialize OpenCV Bridge
		self.bridge = CvBridge()

		curRosPackage = rospkg.RosPack()
		self.curModulePath = curRosPackage.get_path('hello_real_sense')

		# initialize classifier
		print 'initializing the classifier'
		self.imgProcessor = ImageProcessor()

		# set subscriber
		self.imageSubscriber = rospy.Subscriber("/camera/color/image_raw",Image, self.processMyImage, queue_size= 20)

		# set publisher
		self.imagePublisher = rospy.Publisher("/sidewalk_detector/color/image_raw",Image, queue_size= 20)
		self.imgCount = 0
		pass

	def processMyImage(self, data):
		try:
			# get image
			opencvImg = self.bridge.imgmsg_to_cv2(data, "bgr8")

			# original image is upside down (looking at the rviz of the rosbag visualized image)

			imgToProcess = cv2.flip(opencvImg, -1)

			# use classifier to process image
			processedImg = self.imgProcessor.processMyImage(imgToProcess)
			# publish the processed image using the publisher

			self.imagePublisher.publish(self.bridge.cv2_to_imgmsg(processedImg, "bgr8"))
			# cv2.imwrite(self.curModulePath+"/examples/"+str(self.imgCount)+'.jpg', processedImg)
			# self.imgCount += 1
			# cv2.waitKey(5)
		except CvBridgeError as e:
			print e

def main(args=None):
	sideWalkNode = RosImgToOpenCVImageConvertor()
	rospy.init_node('sidewalk_detector', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)