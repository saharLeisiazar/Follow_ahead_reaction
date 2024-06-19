import numpy as np
import math
import random
import gymnasium as gym

def degrees_to_vector(angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)

    # Compute x and y components
    x = math.cos(angle_radians)
    y = math.sin(angle_radians)

    # The vector is already normalized because cos^2 + sin^2 = 1
    return (x, y)

class Environment(gym.Env):
    def __init__(self, target_distance=10, distance_threshold=0.000000001, max_steps=100):
        super(Environment, self).__init__()
        self.target_distance = target_distance
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.human_step_size = 0.5
        self.agent_step_size1 = 0.5
        self.agent_step_size2 = 1
        self.max_x = 50
        self.max_y = 50
        self.action_space = gym.spaces.Discrete(16)
        self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=float)

        self.reset()
        
    def reset(self):
        self.human_position =  np.random.uniform(0, self.max_x, 2) #[2, np.random.uniform(0, self.max_x, 2)]
        #self.human_position[0] = 1
        self.agent_position = np.random.uniform(0, self.max_x, 2) #self.human_position + np.array([0, -1]) #np.random.uniform(4, 5, 1)
        self.agent_position = np.minimum(self.agent_position, np.array([self.max_x, self.max_y]))

        tmp = [0, np.pi/4, np.pi/2, np.pi * 3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4]

        self.human_orientation = 0
        self.agent_orientation = random.choice(tmp)

        self.step_count = 0
        self.history = {'human': [], 'agent': []}  # Track position history for plotting
        return self._get_state()#, {}
    
    def step(self, action):

        self.history['human'].append(self.human_position.copy())
        self.history['agent'].append(self.agent_position.copy())
        
        tmp = [0, np.pi*7/4, np.pi/2, np.pi * 3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*1/4]
        
        self.agent_orientation = tmp[action % 8]

        if action < 8:
            self.agent_position += np.array([
                self.agent_step_size1 * np.cos(self.agent_orientation), 
                self.agent_step_size1 * np.sin(self.agent_orientation)
            ])
        else: 
            self.agent_position += np.array([
                self.agent_step_size2 * np.cos(self.agent_orientation), 
                self.agent_step_size2 * np.sin(self.agent_orientation)])

        self.agent_position = np.minimum(self.agent_position, np.array([self.max_x, self.max_y]))
        self.agent_position = np.maximum(self.agent_position, np.array([0, 0]))

        reward = self._calculate_reward()

        tmp = [0, np.pi/8, -np.pi/8, np.pi/4, -np.pi/4]
        self.human_orientation += random.choice(tmp)

        self.human_position += np.array([self.human_step_size * np.cos(self.human_orientation), self.human_step_size * np.sin(self.human_orientation)])
        self.human_position = np.minimum(self.human_position, np.array([self.max_x, self.max_y]))
        self.human_position = np.maximum(self.human_position, np.array([0, 0]))


        self.step_count += 1
        done = self.step_count >= self.max_steps
        state = self._get_state()
        return state, reward, done, {}
    
    def _get_state(self):
        vector = self.agent_position - self.human_position
        diff = assign_values(vector, self.human_orientation) * 180 / np.pi

        return np.concatenate([self.agent_position - self.human_position, [self.human_orientation], [self.agent_orientation]])
        # Consolidate the state information
        return np.concatenate([
            self.human_position, [self.human_orientation], 
            self.agent_position, [self.agent_orientation]
        ])

    def _calculate_reward(self):
        # Distance from the agent to the human
        #distance = np.linalg.norm(self.agent_position - self.human_position)
        # orientation_diff = np.abs(self.human_orientation - np.arctan2(self.agent_position[1] - self.human_position[1], self.agent_position[0] - self.human_position[0]))
        #
        # # Reward based on distance and orientation
        # if (distance > self.distance_threshold) and (distance < self.target_distance):
        #     if (orientation_diff < np.pi/8):
        #         return self.target_distance * (1 / distance) * (np.pi/8 - orientation_diff) * (8/np.pi)
        #     return 0
        # elif distance > self.target_distance:
        #     return 0
        # else:
        #     return -10 * (self.distance_threshold - distance)
        #
        #
        
        # distance = np.linalg.norm(self.agent_position - self.human_position)

        # if (distance > self.distance_threshold) and (distance < self.target_distance):
        #     return (1 / distance) #* (1 - diff / angular_threshold_degrees)
        # elif distance < self.distance_threshold:
        #     return -1
        # else:
        #     return 0

        vector = self.agent_position - self.human_position

        #angular_threshold_degrees = 2 * np.pi #np.pi / 4
        distance = np.linalg.norm(vector + 1e-8)
        diff = assign_values(vector, self.human_orientation) * 180 / np.pi

        r_d = 0
        if distance > 5 or distance <0.25:
            r_d = -1
        elif distance >0.25 and distance <1:
            r_d = -(1-distance)
        elif distance > 1 and distance <2:
            r_d = 0.5 * (0.5 - np.abs(distance-1.5))
        elif distance > 2 and distance <5:
            r_d = -0.25 * (distance-1)
        else:
            pass

        if diff < 25:
            r_o = 0.5 * ((25 - diff)/25)
        else:
            r_o = -0.25 * diff / 180

        return min(max(r_o + r_d, -1), 1)

        # if (distance > self.distance_threshold) and (distance < self.target_distance) and diff < angular_threshold_degrees:
        #     return self.target_distance * (1 / distance) * (1 - diff / angular_threshold_degrees)
        # elif distance < self.distance_threshold:
        #     return - (self.distance_threshold - distance)
        # else:
        #     return 0



        # vector_to_agent = np.array(self.agent_position) - np.array(self.human_position)

        # # Normalize vectors for angle calculation
        # human_direction_normalized = np.array(human_direction) / np.linalg.norm(human_direction)
        # vector_to_agent_normalized = vector_to_agent / np.linalg.norm(vector_to_agent)

        # # Calculate the cosine of the angle between the human's direction and the vector to the agent
        # cosine_angle = np.dot(human_direction_normalized, vector_to_agent_normalized)

        # angular_threshold_degrees = 10

        # # Calculate the actual angle in degrees
        # angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) * (180 / np.pi)

        # angle = 360 - angle if 360 - angle < angular_threshold_degrees else angle

        # if (distance > self.distance_threshold) and (distance < self.target_distance) and angle < angular_threshold_degrees:
        #     return self.target_distance * (1 / distance) * (1 - angle / angular_threshold_degrees)
        # elif distance < self.distance_threshold:
        #     return -self.target_distance * (self.distance_threshold - distance)
        # else:
        #     return 0

# def assign_values(ref_pos, ref_orientation_radians, objects, angle_threshold_radians):
#     values = np.zeros(len(objects))

#     # Calculate reference direction angle relative to positive x-axis
#     ref_dir_angle = ref_orientation_radians
    
#     for i, obj_pos in enumerate(objects):
#         # Calculate vector from reference to object
#         vector_to_obj = obj_pos - ref_pos
        
#         # Calculate angle of this vector relative to positive x-axis
#         obj_angle = np.arctan2(vector_to_obj[1], vector_to_obj[0])
        
#         # Calculate the angular difference
#         angular_difference = np.abs(obj_angle - ref_dir_angle)
        
#         # Normalize the difference to be within [0, π]
#         angular_difference = np.minimum(angular_difference, 2*np.pi - angular_difference)
        
#         # Check if the object is within the angle threshold
#         if angular_difference <= angle_threshold_radians:
#             # Assign a value inversely proportional to the angle (closer to 0 is more aligned)
#             values[i] = np.cos(angular_difference)
    
#     return values

def assign_values(vector, ref_orientation_radians):
    angle = np.arctan2(vector[1], vector[0])
    if angle < 0:
        angle = 2* np.pi + angle

    diff = np.abs(ref_orientation_radians - angle)

    return diff
    