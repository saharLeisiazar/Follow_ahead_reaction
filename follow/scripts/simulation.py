from sim_human_traj_generate import human_traj_generator
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
    parser.add_argument('--human_vel', type=int, default= 1.)
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
        self.human_traj = human_traj_generator(self.params['human_vel'], self.params['dt'])
        self.human_prob = prob_dist(self.params['human_prob_model_dir'])

        self.params['human_acts'] = self.define_human_actions()
        self.params['robot_acts'] = self.define_robot_actions()
        self.params['robot_vel'] = 1.0
        self.params['robot_vel_fast_lamda'] = 1.7
        self.params['robot_angle'] = 45.
        self.params['human_angle'] = 45.
        self.params['safety_params'] = {"r":0.5, "a":0.25}        

        file_name = 'a2c_len_50_random_walk_new_sahar_rew_small_net.zip'
        model_directory = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/' + file_name 
        self.params['RL_model'] = RL_model()
        self.params['RL_model'].load_model(model_directory, policy='a2c')


    def run(self):
        self.plot_idx = 0
        for traj in self.human_traj:
            sum_reward = 0.
            traj_state = []
            robot_pose = np.array(traj[self.params['human_history_len']-1]) + [1.5,0.,0.]
            for i in range(len(traj)- self.params['human_history_len']):
                # get the human prob destribution
                history_seq = np.array(traj[i:i+self.params['human_history_len']])
                human_prob = self.human_prob.forward(history_seq[:, :2])
                # human_prob = {'left': 1., 'straight': 1., 'right': 1.}

                # expand the tree
                # if i==11:
                #     print('here')
                state = np.array([robot_pose, history_seq[-1]])
                nav_state = navState(params = self.params, state=state, next_to_move= 0)
                node_human = MCTSNode(state=nav_state, params = self.params, parent= None)  
                mcts = MCTS(node_human, human_prob)
                robot_action = mcts.tree_expantion(time.time() + self.params['expansion_time']).action

                robot_pose = self.move_robot(state, robot_action)
                traj_state.append(state)

                ###
                reward = nav_state.calculate_reward(state)
                sum_reward += reward

            self.plot_state(traj_state)
            print('sum_reward:', sum_reward)

        return

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

