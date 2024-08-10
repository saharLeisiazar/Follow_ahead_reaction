#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
import colorsys
from geometry_msgs.msg import Point


class node():
    def __init__(self):
        rospy.init_node('main', anonymous=True)

        self.number_of_points = 60
        self.rainbow_colors = self.generate_rainbow_colors(self.number_of_points)

        rospy.Subscriber('/robot_traj', Marker, self.robot_callback, buff_size=1)
        rospy.Subscriber('/human_traj', Marker, self.human_callback, buff_size=1)
        rospy.Subscriber('/robot_traj_arrow', Marker, self.robot_arr_callback, buff_size=1)
        rospy.Subscriber('/human_traj_arrow', Marker, self.human_arr_callback, buff_size=1)

        self.pub_robot_traj = rospy.Publisher('/robot_traj_mod', Marker, queue_size=1)
        self.pub_human_traj = rospy.Publisher('/human_traj_mod', Marker, queue_size=1)
        self.pub_robot_arrow = rospy.Publisher('/robot_traj_arrow_mod', Marker, queue_size=1)
        self.pub_human_arrow = rospy.Publisher('/human_traj_arrow_mod', Marker, queue_size=1)

        self.id_robot = 0
        self.id_human = 0
        self.id_robot_arrow = 0
        self.id_human_arrow = 0

        self.prev_point = Point()

        print("Initiated")
               
    def generate_rainbow_colors(self, num_points):
        colors = []
        for i in range(num_points):
            hue = i / num_points
            rgb = colorsys.hsv_to_rgb(hue, 1, 1)
            colors.append(rgb)
        return colors

    def robot_callback(self, data):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()

        m.ns = "robot"
        m.id = self.id_robot
        m.type = 1
        m.action = 0

        m.pose.position.x = data.pose.position.x
        m.pose.position.y = data.pose.position.y
        m.pose.orientation.z = 0
        m.pose.orientation.w = 1.

        m.scale.x = .1 
        m.scale.y = .1
        m.scale.z = .1

        m.color.r = self.rainbow_colors[self.id_robot][0]
        m.color.g = self.rainbow_colors[self.id_robot][1]
        m.color.b = self.rainbow_colors[self.id_robot][2]
        m.color.a = 1.

        self.pub_robot_traj.publish(m)

        self.id_robot += 1
        self.id_robot = self.id_robot % self.number_of_points

    def robot_arr_callback(self, data):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()

        m.ns = "robot_arrow"
        m.id = self.id_robot_arrow
        m.type = 0
        m.action = 0

        m.pose.position.x = data.pose.position.x
        m.pose.position.y = data.pose.position.y

        m.pose.orientation.z = data.pose.orientation.z
        m.pose.orientation.w = data.pose.orientation.w

        m.scale.x = .1
        m.scale.y = .08
        m.scale.z = .1

        m.color.r = self.rainbow_colors[self.id_robot_arrow][0]
        m.color.g = self.rainbow_colors[self.id_robot_arrow][1]
        m.color.b = self.rainbow_colors[self.id_robot_arrow][2]
        m.color.a = 1.

        self.pub_robot_arrow.publish(m)

        self.id_robot_arrow += 1
        self.id_robot_arrow = self.id_robot_arrow % self.number_of_points

    # def human_callback(self, data):
    #     m = Marker()
    #     m.header.frame_id = "map"
    #     m.header.stamp = rospy.Time.now()

    #     # m.ns = "human"
    #     # m.id = self.id_human
    #     # m.type = 4
    #     # m.action = 0

    #     m.pose.position.x = data.pose.position.x
    #     m.pose.position.y = data.pose.position.y

    #     m.pose.orientation.z = 0
    #     m.pose.orientation.w = 1.

    #     m.scale.x = .1
    #     m.scale.y = .1
    #     m.scale.z = .1

    #     m.color.r = self.rainbow_colors[self.id_human][0]
    #     m.color.g = self.rainbow_colors[self.id_human][1]
    #     m.color.b = self.rainbow_colors[self.id_human][2]
    #     m.color.a = 1.

    #     self.pub_human_traj.publish(m)  

    #     self.id_human += 1
    #     self.id_human = self.id_human % self.number_of_points

    def human_arr_callback(self, data):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()

        m.ns = "human_arrow"
        m.id = self.id_human_arrow
        m.type = 0 
        m.action = 0

        m.pose.position.x = data.pose.position.x
        m.pose.position.y = data.pose.position.y

        m.pose.orientation.z = data.pose.orientation.z
        m.pose.orientation.w = data.pose.orientation.w

        m.scale.x = .1 
        m.scale.y = .1
        m.scale.z = .1

        m.color.r = self.rainbow_colors[self.id_human_arrow][0]
        m.color.g = self.rainbow_colors[self.id_human_arrow][1]
        m.color.b = self.rainbow_colors[self.id_human_arrow][2]
        m.color.a = 1.

        self.pub_human_arrow.publish(m)

        self.id_human_arrow += 1
        self.id_human_arrow = self.id_human_arrow % self.number_of_points


    def human_callback(self, data):
        ######### human line
        line_strip = Marker()
        line_strip.header.frame_id = 'map'
        line_strip.header.stamp = rospy.Time.now()
        line_strip.ns = "human_lines"
        line_strip.action = 0
        line_strip.pose.orientation.w = 1.0
        line_strip.type = 4
        line_strip.scale.x = 0.05
        line_strip.color.b = 1.0
        line_strip.color.a = 1.0

        line_strip.color.r = self.rainbow_colors[self.id_human][0]
        line_strip.color.g = self.rainbow_colors[self.id_human][1]
        line_strip.color.b = self.rainbow_colors[self.id_human][2]
        line_strip.color.a = 1.

        line_strip.id = self.id_human
        self.id_human += 1
        self.id_human = self.id_human % self.number_of_points

        if self.prev_point.x != 0:
            line_strip.points.append(self.prev_point)

        p = Point()
        p.x = data.pose.position.x
        p.y = data.pose.position.y
        p.z = 0

        line_strip.points.append(p)
        self.pub_human_traj.publish(line_strip)
        self.prev_point = p





if __name__ == '__main__':
    node()
    rospy.spin()

