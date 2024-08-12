#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import os
import statistics
import rospy
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import PoseStamped , PoseArray, PointStamped
from nav_msgs.msg import Odometry
from dynamixel_workbench_msgs.srv import *
from rospy_tutorials.srv import *
import sys
from scipy.spatial.transform import Rotation
# sys.path.append('/home/sahar/catkin_ws/src/Follow_ahead_reaction/rotate_motor/scripts/')
# from camera_dimensions import get_camera_dimensions

# Get camera dimensions
width, height = (1280,1080)
#for testing
save_file = "Human_positions.npy"

class human_traj_prediction():
    def __init__(self):
        print('initializing node')
        rospy.init_node('prediction', anonymous=True)
        
        print('setting up subscriber and publisher')
        rospy.Subscriber('/zed2/zed_node/obj_det/objects', ObjectsStamped, self.predictions_callback)
        print('Subscriber setup for /zed2/zed_node/obj_det/objects')

        rospy.Subscriber('odom', Odometry, self.robot_position_callback)

        self.pub_cur_pose = rospy.Publisher('/person_pose', PoseStamped, queue_size = 1)
        print('Publisher setup for /person_pose')

        self.pub_pred_pose_all = rospy.Publisher('/person_pose_pred_all', PoseArray, queue_size = 1)
        print('Publisher setup for /person_pose_pred_all')

        self.pub_glob_coords = rospy.Publisher('/pub_glob_coords', PointStamped, queue_size=1)
        print('Publisher setup for /pub_glob_coords')


        self.deg_to_res = 1024/90
        self.goal = 0
        self.count = -1
        self.notfound = 0
        self.robot_position = [0,0,0]
        self.human_x = []
        self.human_y = []
        # self.robot_orientation = np.eye(3)
        # self.motor_rotation_angle = 0.0
        
        #For testing
        self.human_global_pos = []

        print('Initialization complete')


    def send_goal(self):

        self.goal = self.goal % 360 # ensures the goal stays in 0-359 range

        print("Sending goal:" , self.goal)
        rospy.wait_for_service('/dynamixel_workbench/dynamixel_command')
        try:
            req = rospy.ServiceProxy('/dynamixel_workbench/dynamixel_command', DynamixelCommand)
            goal_res = int(self.goal* self.deg_to_res)
            print('goal value', goal_res)
            resp1 = req('', 1, 'Goal_Position', goal_res) #update motor rotation angle
            self.motor_rotation_angle = self.goal * (math.pi/180)
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
                    'bb_y_max': obj.bounding_box_2d.corners[2].kp[1],
                    'position': obj.position
                    }

        return feat

    # To get the global coordinates published by the robot
    def robot_position_callback(self, data):
        self.robot_position = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])

        q = data.pose.pose.orientation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        rot_euler = rot.as_euler('xyz', degrees=False)
        self.robot_orientation = rot_euler[2]

    def transform_to_global(self, local_coords):
        local_coords = np.array(local_coords)
        angle = np.deg2rad(self.goal)
        motor_rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                           [np.sin(angle), np.cos(angle), 0],
                                           [0, 0, 1]])
        global_coords = np.dot(motor_rotation_matrix, local_coords) + self.robot_position

        global_coords = global_coords[:2]
        global_ori = self.get_global_ori(global_coords)
        return np.append(global_coords, global_ori)


    def get_global_ori(self, coords):
        self.human_x.append(coords[0])
        self.human_y.append(coords[1])

        if len(self.human_x) > 10:
            self.human_x.pop(0)
            self.human_y.pop(0)

        t = np.arange(len(self.human_x)) #.reshape(-1, 1)
        x = np.array(self.human_x)
        y = np.array(self.human_x)

        coef_x = np.polyfit(t, x, 1)
        coef_y = np.polyfit(t, y, 1)
        p_x = np.poly1d(coef_x)
        p_y = np.poly1d(coef_y)

        new_x = p_x(t[-1]+ 5)
        new_y = p_y(t[-1]+ 5)
        global_ori = np.arctan2(new_y-y[-1], new_x-x[-1])

        return global_ori


    def predictions_callback(self,data):
        self.count += 1
        feat = self.human_data(data)

        ###### person is not detected            
        if feat == {} and self.count % 5 == 0:
            print("Looking for the human ...")
            self.notfound += 1
            print("Not found count: ", self.notfound)
            self.goal = (self.goal + 20) 
            self.send_goal()
            # rospy.sleep(0.3)


        ###### person is detected
        if len(feat) and self.count%3 == 0 :
                # print("Person detected at x: ", feat['x'])
            ###### To keep the human in the middle of the image

                mean_bb = (feat['bb_x_min'] + feat['bb_x_max']) /2
                # print("mean_bb: ", mean_bb)
                # turn = 10
                
                # Define center threshold based on a narrower range
                center_range_percentage = 0.15  # % on either side of the center
                image_center = 1280 / 2
                center_threshold_min = image_center * (1 - center_range_percentage)  # 1280 * (1-0.15)/2 = 576 
                center_threshold_max = image_center * (1 + center_range_percentage)  # 1280 * (1+0.15)/2 = 704

                angle = 15
                if mean_bb < center_threshold_min:
                    turn = min(angle, center_threshold_min - mean_bb)
                    self.goal += turn
                    print("rotating ", turn , " deg")
                    self.send_goal()   

                elif mean_bb > center_threshold_max:
                    turn = min(angle, mean_bb - center_threshold_max)
                    self.goal -= turn
                    print("rotating -", turn , " deg")
                    self.send_goal() 


                # transform the human position to global coordinates
                # human_position_global = self.transform_to_global(feat['position'])
                # print("Human position in global coordinates: ", human_position_global)

                # for testing: store human pos for plotting and testing
                # self.human_global_pos.append(human_position_global.tolist())
                # # print("current global position shage: ", np.array(self.human_global_pos).shape)
                # self.save_positions()

                # Publish the current human position
                #point = PointStamped()
                #point.header.stamp = rospy.Time.now()
                #point.header.frame_id = 'map'
                #point.point.x = human_position_global[0]
                #point.point.y = human_position_global[1]
                #point.point.z = human_position_global[2]
                #self.pub_glob_coords.publish(point)

    #For testing
    def save_positions (self):
        np.save(save_file, np.array(self.human_global_pos))
        print(f"saved to {save_file}")

    # For Testing: To plot the global position of human for testing
    def plot_human_positions(self):
        # global_positions = np.array(self.human_global_pos)
        if os.path.exists(save_file):
            global_positions = np.load(save_file)
        print(f"Global position shape: {global_positions.shape}")
        x_coords = global_positions[:,0]
        y_coords = global_positions[:,1]
        z_coords = global_positions[:,2]

        plt.figure()
        plt.plot(x_coords, y_coords, 'o-', label='Human Trajectory')
        plt.title("Human Position in Global Coordinates")
        plt.xlim(-6,6)
        plt.ylim(-6,6)
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    print("Starting the human trejectory prediction node")
    human_traj_prediction()
    rospy.spin()   

    # human_traj_prediction().plot_human_positions() 
