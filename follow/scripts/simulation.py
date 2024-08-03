from sim_human_traj_generate import human_traj_generator as human_smooth_traj
from sim_human_traj_generate_sudden_move import human_traj_generator as human_sudden_traj
from human_prob_dist import prob_dist, LSTMModel2D
from nodes import MCTSNode
from search import MCTS
from navi_state import navState
from RL_interface import RL_model

import time
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=bool, default= True)
    parser.add_argument('--expansion_time', type=int, default= 0.15)
    parser.add_argument('--gamma', type=float, default= 0.90)
    parser.add_argument('--human_vel', type=int, default= 0.7)
    parser.add_argument('--dt', type=int, default= 0.2)
    parser.add_argument('--human_history_len', type=int, default= 15)
    parser.add_argument('--human_prob_model_dir', type=str, default= "/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/human_prob.pth")
    args = parser.parse_args()
    params = vars(args)

    ################## start ##################
    tree = Tree(params)
    tree.run()


class Tree(object):
    def __init__(self, params):
        #parameters
        self.params = params 
        # self.human_traj = human_smooth_traj(self.params['human_vel'], self.params['dt'])
        self.human_traj = human_sudden_traj(self.params['human_vel'], self.params['dt'])
        self.human_prob = prob_dist(self.params['human_prob_model_dir'])

        self.params['human_acts'] = self.define_human_actions()
        self.params['robot_acts'] = self.define_robot_actions()
        self.params['robot_vel'] = 0.7
        self.params['robot_vel_fast_lamda'] = 2.
        self.params['robot_angle'] = 45.
        self.params['human_angle'] = 10.
        self.params['safety_params'] = {"r":0.5, "a":0.25}        

        file_name = 'multiply_rewards_1.zip'
        model_directory = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/' + file_name 
        self.params['RL_model'] = RL_model()
        self.params['RL_model'].load_model(model_directory, policy='a2c')


    def run(self):
        self.plot_idx = 0
        robot_init_pose_list = [[1.3,0.,0.], [1.3, 0.5, 0], [1.3, -1., 0], [1.,1, 0.], [1.3, -0.5, 0.4], [1., -0.5, 0.5]]
        if os.path.exists('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/summary.txt'):
            os.remove('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/summary.txt')
            
        for robot_init in robot_init_pose_list:
            for traj in self.human_traj:
                sum_reward = 0.
                dist_list = []
                angle_list = []
                traj_state = []
                robot_pose = np.array(traj[self.params['human_history_len']-1]) + robot_init
                for i in range(len(traj)- self.params['human_history_len']):
                    # get the human prob destribution
                    history_seq = np.array(traj[i:i+self.params['human_history_len']])
                    human_prob = self.human_prob.forward(history_seq[:, :2])
                    # human_prob = {'left': 0.1, 'straight': 0.8, 'right': 0.1}

                    # expand the tree
                    # if i==11:
                    #     print('here')
                    state = np.array([robot_pose, history_seq[-1]])
                    # state = np.array([[-0.47, -1.04, 0.03],[-2.5, -1.06, -0.02]])
                    nav_state = navState(params = self.params, state=state, next_to_move= 0)
                    node_human = MCTSNode(state=nav_state, params = self.params, parent= None)  
                    mcts = MCTS(node_human, human_prob)
                    robot_action = mcts.tree_expantion(time.time() + self.params['expansion_time']).action

                    robot_pose = self.move_robot(state, robot_action)
                    traj_state.append(state)

                    ###
                    reward = nav_state.calculate_reward(state)
                    sum_reward += reward
                    dist, angle = self.compute_mean_dist_angle(state)
                    dist_list.append(dist)
                    angle_list.append(angle)

                self.plot_state(traj_state)
                print('sum_reward:', sum_reward)
                print('mean_dist:', np.mean(dist_list), "std_dist:", np.std(dist_list))
                print('mean_angle:', np.mean(angle_list), "std_angle:", np.std(angle_list))
                # Write sum_reward and mean_dist to a text file
                with open('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/summary.txt', 'a') as file:
                    file.write(f'robot_init_pose: {robot_init}\n')
                    file.write(f'sum_reward: {sum_reward}\n')
                    file.write(f'mean_dist: {np.mean(dist_list)}\n')
                    file.write(f'std_dist: {np.std(dist_list)}\n')
                    file.write(f'mean_angle: {np.mean(angle_list)}\n')
                    file.write(f'std_angle: {np.std(angle_list)}\n')
                    file.write('\n')
                
        return

    def compute_mean_dist_angle(self, state):
        robot_pose = state[0]
        human_pose = state[1]
        dist = np.linalg.norm(robot_pose[:2] - human_pose[:2])
        angle = np.arctan2(robot_pose[1] - human_pose[1], robot_pose[0] - human_pose[0]) - human_pose[2]
        # print('dist:', dist, 'angle:', angle)
        return dist, angle

    def plot_state(self, traj_state):
        traj_state = np.array(traj_state)
        step=int(270/traj_state.shape[0])
        fig, axs = plt.subplots(1, figsize = (12,7))

        robot_m_size = 100
        human_m_size =70
        for i in range(traj_state.shape[0]):
            C = plt.get_cmap("jet")(step*i)
            if i == 0:
                axs.scatter(traj_state[i,0,0], traj_state[i,0,1], color=C, s=robot_m_size, label='Robot', marker='o') #, linestyle=' '
                axs.scatter(traj_state[i,1,0], traj_state[i,1,1], color=C, s=human_m_size, label='Human', marker='x')
            else:
                axs.scatter(traj_state[i,0,0], traj_state[i,0,1], s=robot_m_size,color=C, marker='o')
                axs.scatter(traj_state[i,1,0], traj_state[i,1,1], s=human_m_size,color=C, marker='x')

        axs.legend()
        axs.set_title('Simulation')
        axs.set_xlabel('X')
        axs.set_ylabel('Y')
        axs.axis('equal')
        
        path = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/sim_results'
        if not os.path.exists(path):
            os.makedirs(path)

        fig.savefig(path + '/' + str(self.plot_idx) + '.png')
        self.plot_idx+=1
        return

    def move_robot(self, state, robot_action):
        nav_state = navState(params = self.params, state=state, next_to_move= 0)
        new_state = nav_state.move(robot_action)
        return new_state.state[0]

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



if __name__ == '__main__':
    main()

