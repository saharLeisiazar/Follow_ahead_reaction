import numpy as np
import math 

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
    
    def assign_values(self, vector, ref_orientation_radians):
        angle = np.arctan2(vector[1], vector[0])
        if angle < 0:
            angle = 2* np.pi + angle

        diff = np.abs(ref_orientation_radians - angle)

        return diff
         
    def calculate_reward(self, state):
        distance = np.linalg.norm(state[0,:2] - state[1, :2])

        beta = np.arctan2(state[0,1] - state[1,1] , state[0,0] - state[1,0])   # atan2 (yr - yh  , xr - xh)           
        diff = np.absolute(state[1,2] - beta) * 180/np.pi   #angle between the person-robot vector and the person-heading vector   

        if diff > 180:
            diff = 360 - diff

        r_d = 0
        if distance > 5 or distance <0.5:
            r_d = -1
        elif distance >0.5 and distance <1:
            r_d = -(1-distance)
        elif distance > 1 and distance <2:  
            r_d = 0. #0.5 * (0.5 - np.abs(distance-1.5))
        elif distance > 2 and distance <5:
            r_d = -0.25 * (distance-1)
        else:
            pass

        if diff < 25:
            r_o = 0.5 * ((25 - diff)/25)
        else:
            r_o = -0.25 * diff / 180

        return min(max(r_o + r_d, -1), 1)
    