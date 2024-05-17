#!/usr/bin/env python3
import numpy as np
import time
import sys
sys.path.insert(0, '/home/sahar/Follow-ahead-3/MCTS_reaction/scripts')
from nodes import MCTSNode
from search import MCTS
from navi_state import navState

import message_filters
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
# import torch
# from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
# import time
# from visualization_msgs.msg import Marker

print("workssssss")
# Initial pose of camera with respect to the robot's init pose
# camera2map_trans = [5.8, 1.3, 0]
# camera2map_rot = [0,0, -150*np.pi/180]


class node():
    def __init__(self):
        rospy.init_node('main', anonymous=True)

        self.params= {}
        self.params['robot_vel'] = 1.0
        self.params['robot_vel_fast_lamda'] = 1.7
        self.params['human_vel'] = 1.0
        self.params['dt'] = 0.5
        self.params['robot_angle'] = 45.
        self.params['human_angle'] = 45.
        self.params['safety_params'] = {"r":0.5, "a":0.25}
        self.params['reaction_zone_params'] = {"r":0.7, "a":0.3}
        self.params['human_acts'] = self.define_human_actions()
        self.params['robot_acts'] = self.define_robot_actions()
        self.params['num_expansion'] = 60
        self.stay_bool = True
        self.time = time.time()
        self.freq = 2 #(Hz)
        self.best_action = None

        # rospy.Subscriber("/test", String, self.move_robot, buff_size=1)
        # rospy.Subscriber("person_pose_pred_all", PoseArray, self.human_pose_callback, buff_size=1)

        helmet_sub = message_filters.Subscriber("vicon/helmet_sahar/root", TransformStamped)
        robot_sub = message_filters.Subscriber("vicon/robot_sahar/root", TransformStamped)

        ts = message_filters.TimeSynchronizer([helmet_sub, robot_sub], 10)
        ts.registerCallback(self.vicon_callback)

        # self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size =1)
        # self.pub_goal_vis = rospy.Publisher('/goal_vis', Marker, queue_size =1)
        self.move_robot = rospy.Publisher('/robot/robotnik_base_control/cmd_vel', Twist, queue_size = 1)

        # file_name = 'RL_model.pt'
        # model_directory = '/path_to_model/' + file_name 
        # model = torch.load(model_directory)
        # self.MCTS_params['model'] = model
 
        

    def vicon_callback(self, helmet, robot):

        ######## robots pose
        orien = robot.transform.rotation
        r = R.from_quat([orien.w, orien.x, orien.y, orien.z])
        robot_z = r.as_euler('zyx', degrees=False)[2]
        if robot_z > 0:
            robot_z -=np.pi
        else:
            robot_z += np.pi

        robot_p = robot.transform.translation

        ######### human pose
        orien = helmet.transform.rotation
        r = R.from_quat([orien.w, orien.x, orien.y, orien.z])
        human_z = r.as_euler('zyx', degrees=False)[2]
        
        if human_z > 0:
            human_z -=np.pi
        else:
            human_z += np.pi

        human_p = helmet.transform.translation

        ######## define state for MCTS
        state = np.array([[robot_p.x, robot_p.y, robot_z],[human_p.x, human_p.y, human_z]])
        self.move()
        
        if time.time() - self.time > (1./self.freq):
            self.time = time.time()
            self.best_action = self.expand_tree(state) 
            print(self.best_action)

        

    def move(self):
        action = self.best_action

        V = self.params['robot_vel'] 
        if action == "fast_straight" or action == "fast_right" or action == "fast_left":
            V *= self.params['robot_vel_fast_lamda']

        W = self.params['robot_angle'] * np.pi / 180 / self.params['dt']
        if action == "straight" or action == "fast_straight":
            W = 0
        elif action == "right" or action == "fast_right":
            W *= -1

        elif action == None:
            V = 0
            W = 0

        t = Twist()
        t.linear.x = V
        t.angular.z = W
        self.move_robot.publish(t)


    def expand_tree(self, state):
        # print("current state: ", self.state)

        if not self.stay():

            nav_state = navState(params = self.params, state=state, next_to_move= 0)
            node_human = MCTSNode(state=nav_state, params = self.params, parent= None)  
            mcts = MCTS(node_human)

            t1 = time.time()
            return  mcts.tree_expantion().action
            print("expansion time: ", time.time()-t1)
            
            # print("leaf node: ", leaf_node.state.state)
            # self.generate_goal(leaf_node.state.state)
            # self.time += self.updateGoal
        else:
            print("Waiting ...")


    def stay(self):
        return False
        if self.stay_bool:
            s = self.state
            D = np.linalg.norm(s[0,:2]- s[1, :2]) #distance_to_human
            beta = np.arctan2(s[0,1] - s[1,1] , s[0,0] - s[1,0])   # atan2 (yr - yh  , xr - xh)           
            alpha = np.absolute(s[1,2] - beta) *180 /np.pi  #angle between the person-robot vector and the person-heading vector         
    
            if D > 2.2:
                return True

            self.stay_bool = False
            return False


    def define_human_actions(self):             
        actions = {0: "left",
                1: "right",
                2: "straight"}
               
        return actions
    
    def define_robot_actions(self):       
        actions = {0: 'fast_left',
                   1: 'fast_right',
                   2: 'fast_straight',
                   3: 'left',
                   4: 'right',
                   5: 'straight'}
             
        return actions




    # def odom_callback(self,data):
    #     robot_p = data.pose.pose.position
    #     robot_o = data.pose.pose.orientation
    #     yaw = R.from_quat([0, 0, robot_o.z, robot_o.w]).as_euler('xyz', degrees=False)[2]
    #     self.state[0,:] = [robot_p.x , robot_p.y, yaw]


    # def human_pose_callback(self, data):
    #     human_traj = np.zeros((7,3))

    #     for i in range(len(data.poses)):
    #         human_traj[i,:] = [data.poses[i].position.x,
    #                             data.poses[i].position.y,
    #                             data.poses[i].orientation.z]

    #     rot_yaw = camera2map_rot[2]
    #     R_tran = [[np.cos(rot_yaw), -1*np.sin(rot_yaw), 0], [np.sin(rot_yaw), np.cos(rot_yaw), 0], [0, 0, 1]]

    #     Traj_list = np.zeros((7,3))
    #     for i in range(human_traj.shape[0]):
    #         new_elem= list(np.add(np.dot(R_tran , human_traj[i, :]), camera2map_trans[:2]+[rot_yaw]))
    #         if new_elem[2] > np.pi:  new_elem[2] -= 2*np.pi
    #         if new_elem[2] < -np.pi:  new_elem[2] += 2*np.pi

    #         Traj_list[i] = new_elem
    #     self.state[1,:] = Traj_list[0,:]

    #     m = Marker()
    #     m.header.frame_id = 'map'
    #     m.header.stamp = rospy.Time.now()
    #     m.pose.position.x = Traj_list[0,0]
    #     m.pose.position.y = Traj_list[0,1]
    #     m.pose.orientation.z = R.from_euler('z', Traj_list[0,2], degrees=False).as_quat()[2]
    #     m.pose.orientation.w = R.from_euler('z', Traj_list[0,2], degrees=False).as_quat()[3]
    #     m.scale.x = 0.7
    #     m.scale.y = 0.1
    #     m.color.a = 1
    #     m.color.g = 255

    #     self.pub_human_pose.publish(m)

    #     Traj_dic={}
    #     Traj_dic['traj'] = Traj_list
        
    #     if time.time() > self.time: 
    #         self.expand_tree([Traj_dic])

    # def generate_goal(self, goal_state):
    #     goal_map_frame = [goal_state[0,0], goal_state[0,1]]
        
    #     print('goal_map_frame', goal_map_frame)

    #     goal = PoseStamped()
    #     goal.header.frame_id = 'map'
    #     goal.header.stamp = rospy.Time.now()
    #     goal.pose.position.x = goal_map_frame[0]
    #     goal.pose.position.y = goal_map_frame[1]

    #     theta = np.arctan2( goal_map_frame[1] - self.state[0,1]  , goal_map_frame[0] - self.state[0,0]  )
    #     theta_quat = R.from_euler('z', theta, degrees=False).as_quat()

    #     goal.pose.orientation.z = theta_quat[2]
    #     goal.pose.orientation.w = theta_quat[3]

    #     self.pub_goal.publish(goal)
  
    #     g = Marker()
    #     g.header.frame_id = 'map'
    #     g.header.stamp = rospy.Time.now()
    #     g.type = 2
    #     g.pose.position.x = goal_map_frame[0]
    #     g.pose.position.y = goal_map_frame[1]
    #     g.scale.x = 0.3
    #     g.scale.y = 0.3
    #     g.color.a = 1
    #     g.color.g = 0
    #     g.color.b = 0.5
    #     g.color.r = 1
    #     self.pub_goal_vis.publish(g)


if __name__ == '__main__':
    node()
    # rospy.sleep(0.5)
    rospy.spin()

