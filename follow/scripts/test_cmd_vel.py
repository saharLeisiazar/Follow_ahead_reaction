#!/usr/bin/env python3

import numpy as np

import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
# import time
from visualization_msgs.msg import Marker, MarkerArray


class node():
    def __init__(self):
        rospy.init_node('main', anonymous=True)

        self.move_robot = rospy.Publisher('/robot_traj', Marker, queue_size = 1)

        self.move()

               

 
    def move(self):


        while not rospy.is_shutdown():
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()

            m.ns = "basic_shapes"
            m.id = 0
            m.type = 1
            m.action = 0

            m.pose.position.x = 0
            m.pose.position.y = -1.
            m.pose.orientation.w = 1.

            m.scale.x = .1
            m.scale.y = .1
            m.scale.z = .1

            m.color.r = 1.0
            m.color.g = 1.
            m.color.b = 0.
            m.color.a = 1.

            self.move_robot.publish(m)




 

if __name__ == '__main__':
    node()
    rospy.spin()

