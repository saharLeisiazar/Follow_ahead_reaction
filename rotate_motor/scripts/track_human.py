#!/usr/bin/env python3

import statistics
import rospy
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import PoseStamped , PoseArray
from dynamixel_workbench_msgs.srv import *
from rospy_tutorials.srv import *
import sys
# sys.path.append('/home/sahar/catkin_ws/src/Follow_ahead_reaction/rotate_motor/scripts/')
# from camera_dimensions import get_camera_dimensions

# Get camera dimensions
width, height = (1280,1080)

class human_traj_prediction():
    def __init__(self):
        print('initializing node')
        rospy.init_node('prediction', anonymous=True)
        
        print('setting up subscriber and publisher')
        rospy.Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped, self.predictions_callback)
        print('Subscriber setup for /zed2/zed_node/obj_det/objects')

        self.pub_cur_pose = rospy.Publisher('/person_pose', PoseStamped, queue_size = 1)
        print('Publisher setup for /person_pose')

        self.pub_pred_pose_all = rospy.Publisher('/person_pose_pred_all', PoseArray, queue_size = 1)
        print('Publisher setup for /person_pose_pred_all')


        self.deg_to_res = 1024/90
        self.goal = 0
        self.count = -1
        self.notfound = 0
        print('Initialization complete')

    def send_goal(self):

        self.goal = self.goal % 360 # ensures the goal stays in 0-359 range

        print("Sending goal:" , self.goal)
        rospy.wait_for_service('/dynamixel_workbench/dynamixel_command')
        try:
            req = rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command', DynamixelCommand)
            goal_res = int(self.goal* self.deg_to_res)
            print('goal value', goal_res)
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
                feat = {
                    'x':obj.position[0],
                    'y':obj.position[1],
                    'bb_x_min': obj.bounding_box_2d.corners[0].kp[0],
                    'bb_x_max': obj.bounding_box_2d.corners[2].kp[0],
                    'bb_y_min': obj.bounding_box_2d.corners[0].kp[1],
                    'bb_y_max': obj.bounding_box_2d.corners[2].kp[1]}

        return feat


    def predictions_callback(self,data):
        self.count += 1

        if self.count % 3 == 0:
            feat = self.human_data(data)
            # print(feat)

            ###### person is not detected
            
            if feat == {}:
                print("Looking for the human ...")
                self.notfound += 1
                print("Not found count: ", self.notfound)
                self.goal = (self.goal + 20) % 370
                self.send_goal()
                # rospy.sleep(0.3)


            ###### person is detected
            if len(feat):
                    print("Person detected at x: ", feat['x'])
                ###### To keep the human in the middle of the image
    
                    mean_bb = (feat['bb_x_min'] + feat['bb_x_max']) /2
                    print("mean_bb: ", mean_bb)
                    # turn = 10
                    
                    # Define center threshold based on a narrower range
                    center_range_percentage = 0.15  # % on either side of the center
                    image_center = 1280 / 2
                    center_threshold_min = image_center * (1 - center_range_percentage)  # 1280 * (1-0.15)/2 = 576 
                    center_threshold_max = image_center * (1 + center_range_percentage)  # 1280 * (1+0.15)/2 = 704


                    if mean_bb < center_threshold_min:
                        turn = min(10, center_threshold_min - mean_bb)
                        self.goal += turn
                        print("rotating ", turn , " deg")
                        self.send_goal()   

                    elif mean_bb > center_threshold_max:
                        turn = min(10, mean_bb - center_threshold_max)
                        self.goal -= turn
                        print("rotating -", turn , " deg")
                        self.send_goal() 


if __name__ == '__main__':
    print("Starting the human trejectory prediction node")
    human_traj_prediction()
    rospy.spin()    
