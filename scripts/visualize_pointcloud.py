#!/usr/bin/env python

import rospy
from tams_bartender_recognition.msg import RecognizedObject
from sensor_msgs.msg import PointCloud2

####################
# global variables
####################
pub = rospy.Publisher("recognized_object_point_cloud", PointCloud2, queue_size=10)


####################
# publish the pointcloud from the incoming object as single pointcloud
####################
def extract_pointcloud_from_msg(recognized_object_msg):
    pub.publish(recognized_object_msg.point_cloud)


####################
# main-function, which start the subscriber and initialize the node
####################
def main():
    rospy.init_node('visualize_segmented_pointcloud')

    rospy.Subscriber("segmented_object", RecognizedObject, extract_pointcloud_from_msg, queue_size = 10)
    rospy.spin()

####################
# starts the main and thus the node 
####################
if __name__ == '__main__':
    main()
