# hello_realsense_camera


Simple ROS package to read data from real_sense_camera bag file and publish images to topics

### NOTE: This only reads the camera data. I have not finished processing the PCL data.

STEPS:

#### First play the bag file recorded
#### Next run the rosnode as $rosrun hello_real_sense video_processing_node.py
#### data is published to topic /sidewalk_detector/color/image_raw


# A Sample Input Image
![Sample input image](https://raw.githubusercontent.com/aceveggie/hello_realsense_camera/master/hello_real_sense/img29_input.jpg)

# A Sample Output Image
![Sample output image](https://raw.githubusercontent.com/aceveggie/hello_realsense_camera/master/hello_real_sense/img29_input.jpg)

# Working example screenshot:
![Working example screenshot](https://raw.githubusercontent.com/aceveggie/hello_realsense_camera/master/hello_real_sense/working_screenshot.png)

