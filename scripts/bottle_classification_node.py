#!/usr/bin/env python


# opencv
import cv2
from cv_bridge import CvBridge, CvBridgeError

# misc
from collections import deque

# ROS
import rospy
import moveit_commander
import tf

# classifier
from label_classifier import label_classifier

# messages
from orbbec_astra_ip.msg import SegmentedBottleArray


#bottle_mesh = "package://orbbec_astra_ip/meshes/bottle.stl"
#bottle_mesh = "../meshes/bottle.stl"
#scene = moveit_commander.PlanningSceneInterface()

# initialize global variables
q = deque([])
br = tf.TransformBroadcaster()

# receive bottles message and put those with images into the queue
def bottle_callback(bottles):
    global q
    for i, bottle in enumerate(bottles.bottles):
        if(len(bottle.image.data) > 0):
            q.append(bottle)

# get bottle label and publish mesh collion object
def consume_bottle(bottle):
    label = classify_bottle(bottle)
    if(label is not None):
        print label
        p = bottle.pose.pose.position
        o = bottle.pose.pose.orientation
        transl = (p.x, p.y, p.z)
        orient = (o.x, o.y, o.z, o.w)
        br.sendTransform(transl, orient, rospy.Time.now(), label, "/surface")
        # this is broken!
        #scene.add_mesh(label, bottle.pose.pose, bottle_mesh, False)

# classify label for bottle message
def classify_bottle(bottle):
    try:
        image = CvBridge().imgmsg_to_cv2(bottle.image, "bgr8")
        return classifier.predict_image(image)
    except CvBridgeError as e:
        print e
    return None

# main function
if __name__=="__main__":
    global q

    #initialize node
    rospy.init_node("sample_collector", anonymous=True)

    # load classifier
    classifier = label_classifier()

    # initiialize bottle subscriber
    bottle_sub = rospy.Subscriber("/segmented_bottles", SegmentedBottleArray, bottle_callback)

    # handle messages in own thread
    while(not rospy.is_shutdown()):
        while(len(q) > 0):
            consume_bottle(q.popleft())
