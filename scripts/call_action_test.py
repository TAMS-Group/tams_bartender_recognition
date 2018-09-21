#!/usr/bin/env python

import sys
import rospy
import actionlib
from tiago_bartender_msgs.msg import *


# main function
if __name__=="__main__":

    try:
        #initialize node
        rospy.init_node("recognize_action_test", anonymous=True)
        client = actionlib.SimpleActionClient('bottle_recognition_action', UpdateBottlesAction)

        client.wait_for_server()

        goal = UpdateBottlesGoal(timeout=rospy.Duration(5.0), stability_threshold=2)

        client.send_goal(goal)

        client.wait_for_result()

        print "Result:", client.get_result()

    except rospy.ROSInterruptException:
        print("program interrupted before completion")

