from ImageProcessor import ImageProcessor
import cv2
import glob

imgProcessor = ImageProcessor()

for eachIndex in range(181):
	img = cv2.imread('/home/jason/catkin_realsense_ws/src/hello_real_sense/scripts/data/img'+str(eachIndex)+'.jpg',1)
	processedImg = imgProcessor.processMyImage(img)
	cv2.imwrite('/home/jason/catkin_realsense_ws/src/hello_real_sense/examples/img'+str(eachIndex)+'.jpg', processedImg)
