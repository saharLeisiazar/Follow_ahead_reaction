#!/usr/bin/env python3

import rospy
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import PoseStamped , PoseArray
from dynamixel_workbench_msgs.srv import *
from rospy_tutorials.srv import *
from camera_dimensions import get_camera_dimensions

# Get camera dimensions
width, height = get_camera_dimensions()

# Use camera dimensions
# print("Camera dimensions: {}x{}".format(width, height))

class human_traj_prediction():
    def __init__(self):
        rospy.init_node('prediction', anonymous=True)
        # self.rate = rospy.Rate(2) # 10hz
        
        rospy.Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped, self.predictions_callback)
        self.pub_cur_pose = rospy.Publisher('/person_pose', PoseStamped, queue_size = 1)
        self.pub_pred_pose_all = rospy.Publisher('/person_pose_pred_all', PoseArray, queue_size = 1)
        print('hello')

        self.deg_to_res = 1024/90
        self.goal = 60
        self.count = -1
        self.last_detection_direction = 'none'

        # the camera has a vision of 110 degrees
        # lets say the human is on the very edge, that is at 110 degree
        # the camera has to rotate 55 degrees to brings human to centre. Thus, max degree
        self.max_rotation_angle = 55 

        # self.p_x_filt =[]
        # self.p_y_filt =[]


    def send_goal(self):
        if self.goal >= 360:
            self.goal -= 360

        elif self.goal <0 :
            self.goal += 360

        print("goal:" , self.goal)
        rospy.wait_for_service('/dynamixel_workbench/dynamixel_command')
        try:
            req = rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command', DynamixelCommand)
            goal_res = int(self.goal* self.deg_to_res)
            resp1 = req('', 1, 'Goal_Position', goal_res)
            # print(resp1.comm_result)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def human_data(self, data):
        feat = {}
        for obj in data.objects:
            if obj.label != 'Person':
                continue
            else:
                feat = {'x':obj.position[0],
                    'y':obj.position[1],
                    'bb_x_min': obj.bounding_box_2d.corners[0].kp[0],
                    'bb_x_max': obj.bounding_box_2d.corners[2].kp[0],
                    'bb_y_min': obj.bounding_box_2d.corners[0].kp[1],
                    'bb_y_max': obj.bounding_box_2d.corners[2].kp[1]}

        return feat

    def predictions_callback(self,data):
        self.count += 1

        # slow down using exhaustive wait ( do we really need it )
        feat = self.human_data(data)
        if(feat != {}):
            self.p_x_bounds.append((feat['bb_x_min'],feat['bb_x_max']))
            self.last_detection_direction = 'left' if feat['x'] < width/2 else 'right'

        #######  person is not detected
        else:
            self.p_x_bounds.append((0,0))

            # detection at edges is not correct so we will try to make smaller turns but 5 turns to cover entire 360
            # avg turn to cover 360 degree and cover most visuals in center pixels (62 degree ?? )
            print("Looking for the human")
            if(self.last_detection_direction == 'left' )
                self.goal -= 62
            else 
                self.goal += 62
            self.send_goal()

        if self.count % 5 == 0:

            median_pair = statistics.median(self.p_x_bounds)
            median_bb_x_min, median_bb_x_max = median_pair 

                
            ###### person is detected
            if len(feat):

                ###### To keep the human in the middle of the image
                mean_bb = (abs(median_bb_x_min) + abs(median_bb_x_max)) /2
                
                # Calculate rotation angle based on mean position
                turn = (mean_bb - width / 2) / (width / 2) * max_rotation_angle
                
                self.goal += turn
                print("rotating -", turn , " deg")
                self.send_goal() 

        # self.rate.sleep()


if __name__ == '__main__':
    human_traj_prediction()
    rospy.spin()    