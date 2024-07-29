import numpy as np
from collections import defaultdict
import util as ptu
from replayBuffer import state_to_obs , normalize


class MCTSNode(object):
    def __init__(self, state, params, parent=None, agent = None):
        self.number_of_visits = 1.
        self.value = 0.
        self.depth = 0
        self.state = state
        self.parent = parent
        self.children = []
        self.params = params
        self.action = ''
        self.tree_id = 0
        self.number_of_reaction = 0


    @property
    def n(self):
        return self.number_of_visits 


    def expand(self):
        a = len(self.children)
        if self.state.next_to_move:
            action = self.params["human_acts"][a]
        else:
            action = self.params["robot_acts"][a]
            
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, self.params, parent=self)
        child_node.action = action

        if not self.state.next_to_move and not self.is_safe_to_pass(next_state):
            child_node = None

        self.children.append(child_node)
        return child_node

    def is_safe_to_pass(self, next_state):

        if not self.close_to_human(next_state):
            if not self.any_obs(self.state.state[0,:2], next_state.state[0,:2]):
                return True

        return False

    def close_to_human(self, next_state):
        r = self.params['safety_params']['r']   # radious of the circle around human
        a = self.params['safety_params']['a']   # displacement from center of the circle
    
        alpha = np.arctan2(next_state.state[0,1]-next_state.state[1,1]  ,next_state.state[0,0]-next_state.state[1,0]  ) #  yr-yh , xr-xh
        alpha = np.absolute (alpha - next_state.state[1,2])

        roots = np.roots([1, -2*a*np.cos(alpha), a*a-r*r])
        d_circle = np.max(roots)
        d_actual = np.linalg.norm(next_state.state[0,:2] - next_state.state[1,:2])

        if d_actual > d_circle:
            return False
        else:
            return True

    def any_obs(self, s, sp):
        # return False
        if self.params['sim']:
            return False
        
        x = int(np.rint((sp[0] - self.params['map_origin_x']) / self.params['map_res']))
        y = int(np.rint((sp[1] - self.params['map_origin_y']) / self.params['map_res']))

        cost = self.params['map_data'][int(x + self.params['map_width'] * y)]

        if cost > 90:
            return True

        # check for occlusion

        # x1_ind = int(np.rint((s[0] - self.params['map_info']['origin_x']) / self.params['map_info']['res']))
        # y1_ind = int(np.rint((s[1] - self.params['map_info']['origin_y']) / self.params['map_info']['res']))
        # x2_ind = int(np.rint((sp[0] - self.params['map_info']['origin_x']) / self.params['map_info']['res']))
        # y2_ind = int(np.rint((sp[1] - self.params['map_info']['origin_y']) / self.params['map_info']['res']))

        # if x2_ind == x1_ind and y2_ind == y1_ind: return False

        # slope = (y2_ind-y1_ind)/(x2_ind-x1_ind+0.00000001)
        # obstacle_count =0
        # s_list_step = np.sign(int((x2_ind-x1_ind)))
        # if s_list_step ==0: s_list_step = 1 
        # x_list = range(x1_ind, x2_ind+1, s_list_step)
        # for x in x_list:
        #     y0 = int(slope*(x-x1_ind) + y1_ind)
        #     for y in range(y0 -2, y0 +3):   
        #         if self.params['map_info']['data'][int(x + self.params['map_info']['width'] * y)] == 100: 
        #             obstacle_count += 1

        # return True if obstacle_count > 2 else False


    def backpropagate(self):
        value = self.value
        parent= self.parent
        pow = 1
              
        while parent:
            parent.number_of_visits +=1
            parent.value += value * self.params['gamma'] ** pow
            parent = parent.parent
            pow +=1


    def is_fully_expanded(self):
        if self.state.next_to_move == 1:
            return len(self.children) == len(self.params["human_acts"])
        else:
            return len(self.children) == len(self.params["robot_acts"])
        
