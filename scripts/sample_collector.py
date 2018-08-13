#!/usr/bin/env python

import sys
import select

#ROS
import rospy

# ROS messages
from pcl_object_recognition.msg import SegmentedObjectArray, SegmentedObject
from sensor_msgs.msg import Image

# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os

# Convenience tool to dump images of object candidates received from the pcl cluster extraction
class sample_collector:

    def __init__(self):
        self.object_sub = rospy.Subscriber("/segmented_objects", SegmentedObjectArray, self.object_callback)
        self.image_pub = rospy.Publisher("/label_samples", Image, queue_size=1)
        self.bridge = CvBridge()
        self.last_image = None
        output = "Press <Enter> to take samples!"
        sample_id = 0
        while(not rospy.is_shutdown()):
            command = raw_input(output) # check for enter pressed
            if(command==""):
                output = self.take_sample(sample_id)
                sample_id+=1
            else:
                output = "Unknown command: " + command

        #while sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        #    line = sys.stdin.readline()
        #    if line:
        #        on_keyboard_input(line)

    def object_callback(self, objects):
        #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        if(objects.count == 1):
            object = objects.objects[0]
            if len(object.image.data):
                self.last_image = object.image
        elif objects.count > 1:
            rospy.logwarn('Got {} objects instead of one.'.format(objects.count))


    def take_sample(self, i):
        ret_msg = "Saving sample to file: "
        image = self.last_image 
        if(image is not None):
            target_dir = 'samples'
            if not os.path.isdir(target_dir):
                if os.path.isfile(target_dir):
                    os.remove(target_dir)
                os.mkdir(target_dir)

            try:
                cv_image = CvBridge().imgmsg_to_cv2(image, "bgr8")
                filename = "object"+str(i)+".png"
                cv2.imwrite(target_dir + '/' + filename, cv_image)
                ret_msg += filename
            except CvBridgeError as e:
                ret_msg = e
        else:
            ret_msg = "Could not find any image data to save!"
        return ret_msg

def main(args):
    rospy.init_node("sample_collector", anonymous=True)
    sc = sample_collector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down!"
    cv2.destroyAllWindows()

if __name__=="__main__":
    print "init sample collector"
    main(sys.argv)
