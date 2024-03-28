#!/usr/bin/env python3

import rospy
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import PoseStamped , PoseArray
from dynamixel_workbench_msgs.srv import *
from rospy_tutorials.srv import *


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

        self.p_x_filt =[]
        self.p_y_filt =[]


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
        
            feat = {'x':obj.position[0],
                 'y':obj.position[1],
                 'bb_x_min': obj.bounding_box_2d.corners[0].kp[0],
                 'bb_x_max': obj.bounding_box_2d.corners[2].kp[0],
                 'bb_y_min': obj.bounding_box_2d.corners[0].kp[1],
                 'bb_y_max': obj.bounding_box_2d.corners[2].kp[1]}

        return feat

    def predictions_callback(self,data):
        self.count += 1

        if self.count % 5 == 0:
            feat = self.human_data(data)
            # print(feat)

            #######  person is not detected
            if feat == {}:
                print("Looking for the human")
                self.goal += 120 
                self.send_goal()


            ###### person is detected
            if len(feat):

                ###### To keep the human in the middle of the image
    
                    mean_bb = (feat['bb_x_min'] + feat['bb_x_max']) /2
                    # print("mean_bb: ", mean_bb)
                    turn = 30
                    
                    if mean_bb < 350:
                        self.goal += turn
                        print("rotating ", turn , " deg")
                        self.send_goal()   

                    elif mean_bb > 650:
                        self.goal -= turn
                        print("rotating -", turn , " deg")
                        self.send_goal() 


            # ########### filtering human's pose
            # self.p_x_filt.append(feat['x'])
            # self.p_y_filt.append(feat['y'])

            # # to filter the noise
            # if len(self.p_x_filt) > 3:
            #     self.p_x_filt.pop(0)
            #     self.p_y_filt.pop(0)

            # x = np.mean(self.p_x_filt)    
            # y = np.mean(self.p_y_filt) 

            # poseArray = PoseArray()
            # poseArray.header.frame_id = 'camera'
            # poseArray.header.stamp = rospy.Time.now()

            # #publishing current and filtered pose of the person
            # pose = PoseStamped()
            # pose.header.frame_id = 'camera'
            # pose.header.stamp = rospy.Time.now()
            # pose.pose.position.x = x
            # pose.pose.position.y = y
            # pose.pose.position.z = 0
            # self.pub_cur_pose.publish(pose)

        # self.rate.sleep()


if __name__ == '__main__':
    human_traj_prediction()
    rospy.spin()    