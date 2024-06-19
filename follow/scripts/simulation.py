from sim_human_traj_generate import human_traj_generator
from human_prob_dist import prob_dist, LSTMModel2D
from nodes import MCTSNode
from search import MCTS
from navi_state import navState
from RL_interface import RL_model

import time
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=bool, default= True)
    parser.add_argument('--num_expansion', type=int, default= 60)
    parser.add_argument('--human_vel', type=int, default= 1.)
    parser.add_argument('--dt', type=int, default= 0.5)
    parser.add_argument('--human_history_len', type=int, default= 5)
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
        self.robot_pose = [0.,0.,0.]
        self.params['human_acts'] = self.define_human_actions()
        self.params['robot_acts'] = self.define_robot_actions()
        self.params['robot_vel'] = 1.0
        self.params['robot_vel_fast_lamda'] = 1.7
        self.params['robot_angle'] = 45.
        self.params['human_angle'] = 45.
        self.params['safety_params'] = {"r":0.5, "a":0.25}        

        file_name = 'a2c_navigation_or0.zip'
        model_directory = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/' + file_name 
        self.params['RL_model'] = RL_model()
        self.params['RL_model'].load_model(model_directory, policy='a2c')

    def run(self):
        for traj in self.human_traj:
            for i in range(len(traj)- self.params['human_history_len']):
                # get the human prob destribution
                history_seq = np.array(traj[i:i+self.params['human_history_len']])
                human_prob = self.human_prob.forward(history_seq[:, :2])

                # expand the tree
                state = np.array([self.robot_pose, history_seq[-1]])
                nav_state = navState(params = self.params, state=state, next_to_move= 0)
                node_human = MCTSNode(state=nav_state, params = self.params, parent= None)  
                mcts = MCTS(node_human, human_prob)
                t = time.time()
                robot_action = mcts.tree_expantion().action
                print('time:', time.time()-t)
                self.move_robot(state, robot_action)


        return

    def move_robot(self, state, robot_action):
        nav_state = navState(params = self.params, state=state, next_to_move= 0)
        new_state = nav_state.move(robot_action)
        self.robot_pose = new_state.state[0]
        return

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

