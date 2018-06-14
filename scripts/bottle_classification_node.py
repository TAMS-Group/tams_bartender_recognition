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
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Quaternion


bottle_mesh_uri = "package://orbbec_astra_ip/meshes/bottle-binary.stl"
bottle_mesh = "../meshes/bottle.stl"
scene = moveit_commander.PlanningSceneInterface()

# initialize global variables
q = deque([])
br = tf.TransformBroadcaster()

bottle_poses = {}

# receive bottles message and put those with images into the queue
def bottle_callback(bottles):
    global q
    for i, bottle in enumerate(bottles.bottles):
        if(len(bottle.image.data) > 0):
            q.append(bottle)

# get bottle label and publish mesh collion object
def consume_bottle(bottle):
    global marker_pub, marker_text_pub
    bottle_id, label = classify_bottle(bottle)
    if(label is not None):
        bp = bottle.pose.pose
        bottle_poses[label] = interpolate(bottle_poses[label], bp, 0.1) if label in bottle_poses else bp
        bp = bottle_poses[label]
        print label
        p = bp.position
        o = bp.orientation
        transl = (p.x, p.y, p.z)
        orient = (o.x, o.y, o.z, o.w)
        br.sendTransform(transl, orient, rospy.Time.now(), label, "/surface")
        # this is broken!
        #scene.add_mesh(label, bottle.pose, "/home/henning/ros/src/orbbec_astra_ip/meshes/bottle.stl", (0.01,0.01,0.01))
        marker = Marker()
        marker.id = bottle_id
        marker.header.frame_id = label
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD

        #marker.type = Marker.CYLINDER
        #marker.scale.x = 0.08
        #marker.scale.y = 0.08
        #marker.scale.z = 0.3
        duration = 4

        marker.type = Marker.MESH_RESOURCE
        #marker.mesh_resource = "/home/henning/ros/src/orbbec_astra_ip/meshes/bottle.stl"
        marker.mesh_resource = bottle_mesh_uri
        marker.lifetime = rospy.Duration(duration)
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        marker.color.a = 1.0
        marker.color.g = 1.0

        marker_pub.publish(marker)

        text = Marker()
        text.id = 100 + bottle_id
        text.header.frame_id = label
        text.header.stamp = rospy.Time.now()
        text.lifetime = rospy.Duration(duration)
        text.action = Marker.ADD
        text.type = Marker.TEXT_VIEW_FACING
        text.text = label
        text.pose.orientation.w = 1.0
        text.pose.position.x = 0.0
        text.pose.position.z = 0.28
        text.color.a = 1.0
        text.color.b = 1.0
        text.color.r = 1.0
        text.color.g = 1.0
        text.scale.x = 0.05
        text.scale.y = 0.05
        text.scale.z = 0.05

        marker_text_pub.publish(text)



# classify label for bottle message
def classify_bottle(bottle):
    try:
        img = CvBridge().imgmsg_to_cv2(bottle.image, "rgb8")
        i, label = classifier.predict_label(img)
        return i, label
    except CvBridgeError as e:
        print e
    return None

def interpolate(p1, p2, fraction):
    p = Pose()
    p.position.x = p1.position.x * (1-fraction) + p2.position.x * fraction
    p.position.y = p1.position.y * (1-fraction) + p2.position.y * fraction
    p.position.z = p1.position.z * (1-fraction) + p2.position.z * fraction
    q1 = [p1.orientation.x, p1.orientation.y, p1.orientation.z, p1.orientation.w]
    q2 = [p2.orientation.x, p2.orientation.y, p2.orientation.z, p2.orientation.w]
    p.orientation = Quaternion(*tf.transformations.quaternion_slerp(q1, q2, fraction))
    return p

# main function
if __name__=="__main__":
    global q, marker_pub, marker_text_pub

    #initialize node
    rospy.init_node("sample_collector", anonymous=True)

    # load classifier
    classifier = label_classifier()

    # initiialize bottle subscriber
    bottle_sub = rospy.Subscriber("/segmented_bottles", SegmentedBottleArray, bottle_callback)

    marker_pub = rospy.Publisher("/bottle_markers", Marker, queue_size=1)
    marker_text_pub = rospy.Publisher("/bottle_label_markers", Marker, queue_size=1)



    # handle messages in own thread
    while(not rospy.is_shutdown()):
        while(len(q) > 0):
            consume_bottle(q.popleft())
