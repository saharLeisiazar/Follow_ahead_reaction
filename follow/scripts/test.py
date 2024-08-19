import numpy as np

mean_dist = np.array([])
std_dist = np.array([])

mean_angle = np.array([])
std_angle = np.array([])

sum_reward = np.array([])

traj_length = 15

with open('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/curr_4.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line == '\n':
            continue
        elif line.split()[0] == "mean_dist:":
            mean_dist = np.append(mean_dist, float(line.split()[1]))
        elif line.split()[0] == "std_dist:":
            std_dist = np.append(std_dist, float(line.split()[1]))
        elif line.split()[0] == "mean_angle:":
            mean_angle = np.append(mean_angle, float(line.split()[1]))
        elif line.split()[0] == "std_angle:":
            std_angle = np.append(std_angle, float(line.split()[1]))
        elif line.split()[0] == "sum_reward:":
            sum_reward = np.append(sum_reward, float(line.split()[1]))


sum_reward /= traj_length

print("sum_reward", np.mean(sum_reward), np.std(sum_reward))
print("mean_dist", np.mean(mean_dist))
print("std_dist", np.mean(std_dist))
print("mean_angle", np.mean(mean_angle))
print("std_angle", np.mean(std_angle))