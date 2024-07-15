import numpy as np
import math 
import matplotlib.pyplot as plt

class navState(object):

    def __init__(self, params, state, next_to_move):
        self.params = params
        self.state = state 
        self.next_to_move = next_to_move  # 0: robot's turn,   1: human's turn

    def nextToMove(self):
        return self.next_to_move

    def move(self, action):
        new_state = self.calculate_new_state(action)
        next_to_move = 0 if self.next_to_move == 1 else 1
        return navState(self.params, new_state, next_to_move) 

    def calculate_new_state(self, action):

        if self.next_to_move == 0: # robot's turn
            ind = 0
            dist = self.params['robot_vel'] * self.params['dt']
            angle = self.params['robot_angle'] * np.pi /180
        else:  # human's turn
            ind = 1
            angle = self.params['human_angle'] * np.pi /180
            dist = self.params['human_vel'] * self.params['dt']

        if action == 'right' or action == 'fast_right':
            angle *= -1.

        if action == 'straight' or action == 'fast_straight':
            angle *=0

        if action == 'fast_straight' or action == 'fast_left' or action == 'fast_right':
            dist *= self.params['robot_vel_fast_lamda']

        new_s = np.copy(self.state)
        new_s[ind,0] = self.state[ind,0] + dist * math.cos(angle + self.state[ind,2])
        new_s[ind,1] = self.state[ind,1] + dist * math.sin(angle + self.state[ind,2])
        new_s[ind,2] = angle + self.state[ind,2]         

        return new_s
    
         
    def calculate_reward(self, state):
        distance = np.linalg.norm(state[0,:2] - state[1, :2])

        beta = np.arctan2(state[0,1] - state[1,1] , state[0,0] - state[1,0])   # atan2 (yr - yh  , xr - xh)           
        diff = np.absolute(state[1,2] - beta) * 180/np.pi   #angle between the person-robot vector and the person-heading vector   

        if diff > 180:
            diff = 360 - diff

        # if diff < 50: # 2 *25
        r_o = 1. * ((25 - diff)/25)
        # else:
        #     r_o = -1.

        if distance > 4 or distance <0.5:
            r_d = -1
        elif distance >=0.5 and distance <=1:
            r_d = -2 * (1-distance)
        elif distance > 1 and distance <=2:  
            r_d = 2. * (0.5 - np.abs(distance-1.5))
        elif distance > 2 and distance <=4:
            r_d = -0.5 * (distance-2)

        ### from [-1,1] to [0,1]
        r_d /= 2
        r_d += 0.500000001
        ### from [-1,1] to [1,3]
        # r_o += 2

        r = r_d + r_o #if r_o > 1 else -1
        return r

  