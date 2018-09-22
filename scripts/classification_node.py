#!/usr/bin/env python


# opencv
import cv2
from cv_bridge import CvBridge, CvBridgeError

# misc
import numpy as np
from collections import deque
import copy

# ROS
import rospy
import moveit_commander
import tf

# classifier
from classifier import label_classifier

# messages
from tams_bartender_recognition.msg import SegmentedObjectArray
from tams_bartender_recognition.msg import RecognizedObject
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Quaternion


bottle_mesh_uri = "package://tams_bartender_recognition/meshes/bottle-binary.stl"
glass_mesh_uri = "package://tams_bartender_recognition/meshes/glass-binary.stl"

bottle_mesh = "../meshes/bottle.stl"
scene = moveit_commander.PlanningSceneInterface()



text_marker = Marker()
bottle_marker = Marker()
glass_marker = Marker()

bottle_poses = {}

candidates = []

# receive segmented objects message and put those with images into the queue
def object_callback(objects):
    global q
    for i, obj in enumerate(objects.objects):
        if(len(obj.image.data) > 0):
            q.append(obj)

def distanceOf(pose1, pose2):
    p1 = pose1.position
    p2 = pose2.position
    return np.linalg.norm([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])

def consume_candidate(candidate):
    global labels, candidates, br, surface_frame
    prediction = make_prediction(candidate)
    pose = candidate.pose.pose
    if prediction is not None:

        # remove old candidates
        candidates = filter(lambda (p,pr,t,c): (rospy.Time.now() - rospy.Duration(2.5)) < t, candidates)

        # find closest candidates
        closest_i = None
        min_dist = float("inf")
        for i, (c_pose, c_pred, c_time, count) in enumerate(candidates):
            dist = distanceOf(pose, c_pose)
            if closest_i is None or dist < min_dist:
                    min_dist = dist
                    closest_i = i

        # update poses and predictions of closest candidate or insert new one
        if closest_i is not None and min_dist < 0.1:
            (c_pose, c_pred, c_time, c_count) = candidates[closest_i]
            w = 0.2
            new_pred = ( w * prediction + (1-w ) * c_pred )
            #new_pred = prediction + c_pred
            new_pose = interpolate(c_pose, pose, 0.2)
            candidates[closest_i] = (new_pose, new_pred, rospy.Time.now(), c_count+1)
        else:
            candidates.append((pose, prediction, rospy.Time.now(), 1))

        # find candidate-label matching
        matching = [None for _ in labels]
        candidates_copy = copy.deepcopy(candidates)
        for i in range(len(candidates)):
            matching = find_match(matching, candidates_copy, i)

        # publish matched objects
        for i, (label, match) in enumerate(zip(labels, matching)):
            if match is not None and match[1] > 0.8:
                (pose, pred, time, count) = candidates[match[0]]
                if(count > 2):
                    obj_height = 0.12 if label.startswith('glass') else 0.28
                    pose.position.z = 0.5*obj_height
                    p = pose.position
                    o = pose.orientation
                    transl = (p.x, p.y, p.z)
                    orient = (o.x, o.y, o.z, o.w)
                    #br.sendTransform(transl, orient, rospy.Time.now(), label, surface_frame)
                    publish_markers(i, label, surface_frame, pose, 1)

                    ro = RecognizedObject()
                    ro.id = label
                    ro.pose.pose.position = p
                    ro.pose.pose.orientation = o
                    ro.pose.pose.position.z = 0.5*obj_height
                    ro.pose.header.frame_id = surface_frame
                    publish_recognized_objects(ro)


def find_match(matching, candidates, i):
    (pose, pred, time, count) = candidates[i]
    argmax = np.argmax(pred)
    confidence = pred[argmax]

    # no confidence
    if confidence > 0.5:

        # check for existing match
        match = matching[argmax]

        # if no match exists, insert candidate
        if match is None:
            matching[argmax] = (i, confidence)

        # else we match the more confident and rematch the weaker one
        else:
            # check which match has higher confidence
            (m_i, m_confidence) = match
            replace_match = confidence > m_confidence
            stronger = (i, confidence) if replace_match else match
            weaker = match if replace_match else (i, confidence)

            # match the stronger one
            matching[argmax] = stronger

            # find a new match for the weaker one
            (w_p, w_pr, w_t, w_c) = candidates[weaker[0]]
            w_pr[argmax] = 0.0
            candidates[weaker[0]] = (w_p, w_pr, w_t, w_c)
            matching = find_match(matching, candidates, weaker[0])

    return matching


# get bottle label and publish mesh collion object
#def consume_bottle(bottle):
#    global marker_pub, marker_text_pub, text_marker, br
#    bottle_id, label = classify_bottle(bottle)
#    if(label is not None):
#        obj_height = 0.12 if label.startswith('glass') else 0.3
#        bp = bottle.pose.pose
#        bp.position.z = 0.5*obj_height;
#        #bottle_poses[label] = interpolate(bottle_poses[label], bp, 0.1) if label in bottle_poses else bp
#        #bp = bottle_poses[label]
#        print label
#        p = bp.position
#        o = bp.orientation
#        transl = (p.x, p.y, p.z)
#        orient = (o.x, o.y, o.z, o.w)
#        br.sendTransform(transl, orient, rospy.Time.now(), label, "/surface")
#        # this is broken!
#        #scene.add_mesh(label, bottle.pose, "/home/henning/ros/src/pcl_bottle_recognition/meshes/bottle.stl", (0.01,0.01,0.01))
#
#        duration = 4
#
#        publish_markers(bottle_id, label, duration)

def publish_markers(marker_id, label, frame, pose, duration):
    publish_object_marker(marker_id, label, frame, pose, duration)
    publish_text_marker(marker_id + 100, label, frame, pose, duration, 0.12 if label.startswith('glass') else 0.28)

def publish_object_marker(marker_id, label, frame, pose, duration):
    global glass_marker, bottle_marker, marker_pub
    marker = glass_marker if label.startswith('glass') else bottle_marker
    marker.id = marker_id
    marker.header.frame_id = frame
    marker.header.stamp = rospy.Time.now()
    marker.pose = pose
    marker.lifetime = rospy.Duration(duration)
    marker_pub.publish(marker)

def publish_text_marker(marker_id, label, frame, pose, duration, z_pos):
    global marker_text_pub, text_marker
    text_marker.id = 100 + marker_id
    text_marker.header.stamp = rospy.Time.now()
    text_marker.header.frame_id = frame
    text_marker.text = label
    text_marker.pose = pose
    text_marker.pose.position.z = z_pos 
    text_marker.lifetime = rospy.Duration(duration)
    marker_text_pub.publish(text_marker)

def publish_recognized_objects(ro):
    global recognized_object_pub
    recognized_object_pub.publish(ro)

def init_text_marker():
    global text_marker
    text_marker.action = Marker.ADD
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.pose.orientation.w = 1.0
    text_marker.pose.position.x = 0.0
    text_marker.color.a = 1.0
    text_marker.color.b = 1.0
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.scale.x = 0.05
    text_marker.scale.y = 0.05
    text_marker.scale.z = 0.05

def init_glass_marker():
    global glass_marker
    glass_marker = get_mesh_marker(glass_marker, glass_mesh_uri)

def init_bottle_marker():
    global bottle_marker
    bottle_marker = get_mesh_marker(bottle_marker, bottle_mesh_uri)

def get_mesh_marker(marker, mesh_resource):
    marker.type = Marker.MESH_RESOURCE
    marker.action = Marker.ADD
    marker.mesh_resource = mesh_resource
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.a = 1.0
    marker.color.g = 1.0
    return marker



# classify label for bottle message
#def classify_bottle(bottle):
#    try:
#        img = CvBridge().imgmsg_to_cv2(bottle.image, "rgb8")
#        i, label = classifier.predict_label(img)
#        return i, label
#    except CvBridgeError as e:
#        print e
#    return None

# classify label for bottle message
def make_prediction(candidate):
    try:
        img = CvBridge().imgmsg_to_cv2(candidate.image, "rgb8")
        return np.array(classifier.predict_image(img))
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
    global q, marker_pub, marker_text_pub, recognized_object_pub, labels, br, surface_frame 

    q = deque([])

    #initialize node
    rospy.init_node("sample_collector", anonymous=True)

    # initialize global variables
    br = tf.TransformBroadcaster()
    object_group = rospy.get_param("~object_group")
    version = rospy.get_param("~version")
    surface_frame = rospy.get_param("~surface_frame", "/surface")

    # load classifier
    classifier = label_classifier(object_group, version)
    labels = classifier.get_labels()

    # initiialize bottle subscriber
    object_sub = rospy.Subscriber("/segmented_objects", SegmentedObjectArray, object_callback)

    init_bottle_marker()
    init_glass_marker()
    init_text_marker()

    marker_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    marker_text_pub = rospy.Publisher("/object_label_markers", Marker, queue_size=1)
    recognized_object_pub = rospy.Publisher("/object_poses", RecognizedObject, queue_size=1)



    # handle messages in own thread
    while(not rospy.is_shutdown()):
        while(len(q) > 0):
            #consume_bottle(q.popleft())
            consume_candidate(q.popleft())
