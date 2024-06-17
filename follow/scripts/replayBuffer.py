import numpy as np

def state_to_obs(s, next_to_move):

    if next_to_move == 1:   ### will be fed to robot model
        gamma = np.arctan2(s[2][1]-s[1][1], s[2][0]-s[1][0]) - s[1][2]
        dis_to_rob = np.linalg.norm(s[1,:2] - s[0, :2])
        angle_to_robot = np.arctan2(s[0,1]-s[1,1], s[0,0]-s[1,0])
        angle = s[1,2] - angle_to_robot
        beta = s[0,2] - angle_to_robot
        return np.array([gamma, dis_to_rob, angle, beta])
    
    else:
        deltaP = s[1, :] - s[0, :]
        gamma = np.arctan2((s[2,1]-s[1,1]) , (s[2,0]-s[1,0])) -s[1,2]
        return np.append(deltaP, gamma)

def normalize(data, mean_std, next_to_move):

    if next_to_move == 1:
        mean = mean_std['human_mean']
        std = mean_std['human_std']
    else:
        mean = mean_std['robot_mean']
        std = mean_std['robot_std']

    return (data-mean)/ std