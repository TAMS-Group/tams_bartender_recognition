#!/usr/bin/env python

import rospy
from tams_bartender_recognition.msg import RecognizedObject
from sensor_msgs.msg import PointCloud2

####################
# global variables
####################
pub = rospy.Publisher("classified_object", RecognizedObject,  queue_size=10)


####################
# fills the pointcloud with label
####################
def classifies_object_in_pointcloud(recognized_object_msg):
    recognized_object_msg.class_label = rospy.get_param("~class", "011_banana")
    pub.publish(recognized_object_msg)


def main():
    rospy.init_node('pitcher_classifier')

    rospy.Subscriber("segmented_object", RecognizedObject, classifies_object_in_pointcloud, queue_size = 10)
    rospy.spin()


if __name__ == '__main__':
    main()
