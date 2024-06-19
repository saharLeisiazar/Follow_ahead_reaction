
import numpy as np
import pandas as pd


dist = 0.5
#########################
# S shape
traj = [[0,0,0.]]
for i in range(1,50):
    if i <10 :
        turn = 5 * np.pi/180
    elif i < 30:
        turn = -5 * np.pi/180
    else:    
        turn = 5 * np.pi/180

    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/s.csv", traj,  
              delimiter = ",")


################
# right turn
turn = -7 * np.pi/180
traj = [[0,0,0.]]

for i in range(1, 15):
    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/right.csv", traj,  
              delimiter = ",")
################
# left turn
turn = 7 * np.pi/180
traj = [[0,0,0.]]
for i in range(1, 15):
    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/left.csv", traj,  
              delimiter = ",")

#####################
# straight trajectory
traj = [[0,0,0.]]
for i in range(1, 10):
    x = traj[i-1][0] + dist 
    traj.append([x,0.,0.])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/straight.csv", traj,  
              delimiter = ",")

################
# left U shape
traj = [[0,0,0.]]
for i in range(1,30):
    if i <5 or i>25:
        turn = 0
    else:
        turn = 8.5 * np.pi/180

    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/left_U.csv", traj,  
              delimiter = ",")

################
# right U shape
traj = [[0,0,0.]]
for i in range(1,30):
    if i <5 or i>25:
        turn = 0
    else:
        turn = -8.5 * np.pi/180

    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/right_U.csv", traj,  
              delimiter = ",")



################
# sharp right turn
turn = -15 * np.pi/180
traj = [[0,0,0.]]

for i in range(1, 15):
    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/right_15.csv", traj,  
              delimiter = ",")
################
# sharp left turn
turn = 15 * np.pi/180
traj = [[0,0,0.]]
for i in range(1, 15):
    x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    theta = turn + traj[i-1][2]
    traj.append([x,y,theta])

traj = np.array(traj)
traj = traj[:, :2]
np.savetxt("../include/trajectories/left_15.csv", traj,  
              delimiter = ",")