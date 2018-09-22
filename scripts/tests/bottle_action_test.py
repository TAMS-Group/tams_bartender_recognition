#!/usr/bin/env python

import sys
import rospy
import actionlib
from tiago_bartender_msgs.msg import *


# main function
if __name__=="__main__":

    try:
        #initialize node
        rospy.init_node("detect_bottles_test", anonymous=True)
        client = actionlib.SimpleActionClient('detect_bottles_action', DetectBottlesAction)

        client.wait_for_server()

        goal = DetectBottlesGoal(timeout=rospy.Duration(5.0), stability_threshold=2)

        client.send_goal(goal)

        client.wait_for_result()

        print "Result:", client.get_result()

    except rospy.ROSInterruptException:
        print("program interrupted before completion")

