from human_prob_dist import prob_dist, LSTMModel2D
from RL_interface import RL_model
import numpy as np
import time
from nodes import MCTSNode
from search import MCTS
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
        self.params['reaction_zone_params'] = {"r":0.8, "a":0.3}
        self.params['human_acts'] = self.define_human_actions()
        self.params['robot_acts'] = self.define_robot_actions()
        self.params['expansion_time'] = 0.15
        self.params['sim'] = False
        self.stay_bool = True
        self.time = time.time()
        self.freq = 5 #(Hz)
        self.best_action = None

        human_prob_model_dir = "/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/human_prob.pth"
        self.human_prob = prob_dist(human_prob_model_dir)
        self.human_history = []
        self.human_history_length = 15
        self.marker_id = 0
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback, buff_size=1)
        # rospy.Subscriber("/map", OccupancyGrid, self.costmap_callback, buff_size=1)

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
        print("Initiated")
               
        self.theta_thr = 20 * np.pi / 180


    def helmet_callback(self, helmet):
        ######## robots pose
        robot_x = self.robot_x
        robot_y = self.robot_y
        robot_z = self.robot_z

        ######### human pose
        orien = helmet.transform.rotation
        
        r = R.from_quat([orien.w, orien.x, orien.y, orien.z])
        human_z = r.as_euler('zyx', degrees=False)[2]
        # human_z *= -1
        human_z -= np.pi/2
        
        if human_z < -np.pi:
            human_z +=2*np.pi
        if human_z > np.pi:
            human_z -= 2*np.pi

        human_z = (np.abs(human_z)//self.theta_thr) *self.theta_thr * np.sign(human_z)

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
                human_prob = self.human_prob.forward(self.human_history)
                # print("human_prob: ", human_prob)
                human_prob = {'left': 0.3, 'straight': 1., 'right': 0.3}

                self.best_action = self.expand_tree(state, human_prob) 
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
        robot_z -= np.pi/2
        
        if robot_z < -np.pi:
            robot_z +=2*np.pi
        if robot_z > np.pi:
            robot_z -= 2*np.pi

        # robot_z = (np.abs(robot_z)//self.theta_thr) *self.theta_thr * np.sign(robot_z)

        robot_p = robot.transform.translation
        theta = 90 * np.pi / 180
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        [robot_x, robot_y] = np.dot(rot, [robot_p.x, robot_p.y])

        self.robot_z = robot_z
        self.robot_x = robot_x
        self.robot_y = robot_y

    def expand_tree(self, state, human_prob=None):
        if not self.stay(state):
            nav_state = navState(params = self.params, state=state, next_to_move= 0)
            node_human = MCTSNode(state=nav_state, params = self.params, parent= None)  
            mcts = MCTS(node_human, human_prob)
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
    
            if D > 1.8:
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
        print("costmap received")

        self.params['map_origin_x'] = data.info.origin.position.x
        self.params['map_origin_y'] = data.info.origin.position.y
        self.params['map_res'] = data.info.resolution
        self.params['map_data'] = data.data
        self.params['map_width'] = data.info.width


if __name__ == '__main__':
    node()
    rospy.spin()

