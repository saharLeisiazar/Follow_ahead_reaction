#!/usr/bin/env python3
# import sys
# sys.path.append('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/scripts')
from human_prob_dist import prob_dist, LSTMModel2D
from RL_interface import RL_model
import numpy as np
import time
from nodes_prev_work import MCTSNode
from search_prev_work import MCTS
from navi_state import navState
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker


class node():
    def __init__(self):
        rospy.init_node('main', anonymous=True)

        self.params= {}
        self.params['robot_vel'] = 0.6
        self.params['robot_vel_fast_lamda'] = 1.5
        self.params['human_vel'] = 0.6
        self.params['dt'] = 0.2
        self.params['gamma'] = 0.9
        self.params['robot_angle'] = 45.
        self.params['human_angle'] = 10.
        self.params['safety_params'] = {"r":0.5, "a":0.25}
        self.params['reaction_zone_params'] = {"r":0.7, "a":0.3}
        self.params['human_acts'] = self.define_human_actions()
        self.params['robot_acts'] = self.define_robot_actions()
        self.params['expansion_time'] = 0.15
        self.params['sim'] = False
        self.stay_bool = True
        self.time = time.time()
        self.freq = 5 #(Hz)
        self.best_action = None

        self.human_history = []
        self.human_history_length = 10
        self.marker_id = 0
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback, buff_size=1)
        rospy.Subscriber("vicon/helmet_sahar/root", TransformStamped, self.helmet_callback, buff_size=1)
        rospy.Subscriber("vicon/robot_sahar/root", TransformStamped, self.robot_callback, buff_size=1)

        self.move_robot = rospy.Publisher('/robot/robotnik_base_control/cmd_vel', Twist, queue_size = 1)
        self.pub_robot_traj = rospy.Publisher('/robot_traj', Marker, queue_size = 1)
        self.pub_human_traj = rospy.Publisher('/human_traj', Marker, queue_size = 1)
        self.pub_robot_arrow = rospy.Publisher('/robot_traj_arrow', Marker, queue_size = 1)
        self.pub_human_arrow = rospy.Publisher('/human_traj_arrow', Marker, queue_size = 1)

        file_name = 'multiply_rewards_1.zip'
        model_directory = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/' + file_name 
        self.params['RL_model'] = RL_model()
        self.params['RL_model'].load_model(model_directory, policy='a2c')

        self.robot_x = 0.
        self.robot_y = 0.
        self.robot_z = 0.0
        print("should be ready")
               

    def helmet_callback(self, helmet):
        ######## robots pose
        robot_x = self.robot_x
        robot_y = self.robot_y
        robot_z = self.robot_z

        ######### human pose
        orien = helmet.transform.rotation        
        r = R.from_quat([orien.w, orien.x, orien.y, orien.z])
        human_z = r.as_euler('zyx', degrees=False)[2]
        human_z *= -1
        human_z -= np.pi/2
        
        if human_z < -np.pi:
            human_z +=2*np.pi
        if human_z > np.pi:
            human_z -= 2*np.pi
 
        human_p = helmet.transform.translation
        theta = 90 * np.pi / 180
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        [human_x, human_y] = np.dot(rot, [human_p.x, human_p.y])
        

        ######## Expanding the tree search
        state = np.array([[robot_x, robot_y, robot_z],[human_x, human_y, human_z]])
        self.move()
        
        if time.time() - self.time > (1./self.freq):
            self.time = time.time()
            self.human_history.append([human_p.x, human_p.y])
            if len(self.human_history) > self.human_history_length:
                self.human_history.pop(0)
                human_future = self.extrapolate(self.human_history)

                self.best_action = self.expand_tree(state, human_future) 
                print()
                print("state: ", state)
                print("action", self.best_action)

                self.pub_marker("robot", self.marker_id, state)
                self.pub_marker("human", self.marker_id, state)
                self.pub_marker("robot", 0, state, arrow=True)
                self.pub_marker("human", 0, state, arrow=True)
                self.marker_id +=1

    def robot_callback(self, robot):
        ######## robots pose
        orien = robot.transform.rotation
        r = R.from_quat([orien.w, orien.x, orien.y, orien.z])
        robot_z = r.as_euler('zyx', degrees=False)[2]
        
        # robot_z *= -1
        robot_z -= np.pi/2
        
        if robot_z < -np.pi:
            robot_z +=2*np.pi
        if robot_z > np.pi:
            robot_z -= 2*np.pi

        # print(robot_z)
        robot_p = robot.transform.translation
        theta = 90 * np.pi / 180
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        [robot_x, robot_y] = np.dot(rot, [robot_p.x, robot_p.y])

        self.robot_z = robot_z
        self.robot_x = robot_x
        self.robot_y = robot_y

    def expand_tree(self, state, human_future=None):
        if not self.stay(state):
            nav_state = navState(params = self.params, state=state, next_to_move= 0)
            node_human = MCTSNode(state=nav_state, params = self.params, parent= None)  
            mcts = MCTS(node_human, human_future)
            best_node = mcts.tree_expantion(time.time()+ self.params['expansion_time'])
            if best_node:
                return  best_node.action
            else:
                return "stop"

        else:
            print("Waiting ...")

    def stay(self, s):
        # return False
        if self.stay_bool:
            D = np.linalg.norm(s[0,:2]- s[1, :2]) #distance_to_human
            beta = np.arctan2(s[0,1] - s[1,1] , s[0,0] - s[1,0])   # atan2 (yr - yh  , xr - xh)           
            alpha = np.absolute(s[1,2] - beta) *180 /np.pi  #angle between the person-robot vector and the person-heading vector         
    
            if D > 1.5:
                return True

            self.stay_bool = False
            return False

    def move(self):
        action = self.best_action

        V = self.params['robot_vel'] 
        if action == "fast_straight" or action == "fast_right" or action == "fast_left":
            V *= self.params['robot_vel_fast_lamda']

        W = self.params['robot_angle'] * np.pi / 180  
        if action == "straight" or action == "fast_straight":
            W = 0
        elif action == "right" or action == "fast_right":
            W *= -1

        elif action == 'stop' or action == None:
            V = 0
            W = 0

        t = Twist()
        t.linear.x = V
        t.angular.z = W
        self.move_robot.publish(t)

    def pub_marker(self, name, id, state, arrow=False):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()

        m.ns = name+"arrow" if arrow else name
        m.id = id
        m.type = 0 if arrow else 1
        m.action = 0

        pose = state[0] if name == "robot" else state[1]
        m.pose.position.x = pose[0]
        m.pose.position.y = pose[1]

        if not arrow:
            m.pose.orientation.z = 0
            m.pose.orientation.w = 1.
        else:
            euler = [0, 0, pose[2]]
            quat = R.from_euler('xyz', euler, degrees=False).as_quat()
            # print(quat)
            m.pose.orientation.z = quat[2]
            m.pose.orientation.w = quat[3]

        m.scale.x = .1 if not arrow else .5
        m.scale.y = .1
        m.scale.z = .1

        m.color.r = 1. if name == "robot" else 0
        m.color.g = 0.
        m.color.b = 0. if name == "robot" else 1
        m.color.a = 1.


        if name=="robot" and not arrow:
            self.pub_robot_traj.publish(m)
        elif name == "robot":
            self.pub_robot_arrow.publish(m)
        elif name == "human" and not arrow:
            self.pub_human_traj.publish(m)
        elif name =="human":    
            self.pub_human_arrow.publish(m)

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

    def costmap_callback(self, data):
        print(1)

        self.params['map_origin_x'] = data.info.origin.position.x - 0.5
        self.params['map_origin_y'] = data.info.origin.position.y - 0.2
        self.params['map_res'] = data.info.resolution
        self.params['map_data'] = data.data
        self.params['map_width'] = data.info.width

        # x = int(np.rint((0. - self.params['map_origin_x']) / self.params['map_res']))
        # y = int(np.rint((-2. - self.params['map_origin_y']) / self.params['map_res']))
        # cost = self.params['map_data'][int(x + self.params['map_width'] * y)]
        # print(cost)


    def extrapolate(self, history_seq):
        t = np.arange(len(history_seq)) #.reshape(-1, 1)
        x = np.array(history_seq)[:, 0]
        y = np.array(history_seq)[:, 1]

        coef_x = np.polyfit(t, x, 1)
        coef_y = np.polyfit(t, y, 1)
        p_x = np.poly1d(coef_x)
        p_y = np.poly1d(coef_y)

        pred = []
        for i in range(1, 10):
            # dt=0.2
            new_x = p_x(t[-1]+ i)
            new_y = p_y(t[-1]+ i)
            theta = np.arctan2(new_y-y[-1], new_x-x[-1])
            pred.append([new_x, new_y, theta])

        return pred
    

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

        ######## Expanding the tree search
        # state = np.array([[0.,0.,0.],[human_p.x, human_p.y, human_z]])
        state = np.array([[robot_p.x, robot_p.y, robot_z],[human_p.x, human_p.y, human_z]])
        self.move()
        
        if time.time() - self.time > (1./self.freq):
            print()
            print("new data")
            print("state: ", state)
            self.time = time.time()
            self.human_history.append([human_p.x, human_p.y])
            if len(self.human_history) > self.human_history_length:
                self.human_history.pop(0)
                # human_prob = self.human_prob.forward(self.human_history)
                human_prob = {'left': 0.1, 'straight': 0.9, 'right': 0.1}

                temp = time.time()
                self.best_action = self.expand_tree(state, human_prob) 
                print("expansion time", time.time()-temp)
                print("action", self.best_action)

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
    rospy.spin()

