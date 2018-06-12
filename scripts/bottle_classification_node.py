#!/usr/bin/env python

# ROS
import rospy

from collections import deque

# classifier
from label_classifier import label_classifier

from orbbec_astra_ip.msg import SegmentedBottleArray

import cv2
from cv_bridge import CvBridge, CvBridgeError

q = deque([])

def classifier_callback(bottles):
    global q
    for i, bottle in enumerate(bottles.bottles):
        if(len(bottle.image.data) > 0):
            q.append(bottle)

def classify_bottle(bottle):
    try:
        image = CvBridge().imgmsg_to_cv2(bottle.image, "bgr8")
        label = classifier.predict_image(image)
        print label
    except CvBridgeError as e:
        print e

if __name__=="__main__":
    global q
    rospy.init_node("sample_collector", anonymous=True)
    classifier = label_classifier()
    bottle_sub = rospy.Subscriber("/segmented_bottles", SegmentedBottleArray, classifier_callback)
    while(not rospy.is_shutdown()):
        while(len(q) > 0):
            classify_bottle(q.popleft())

