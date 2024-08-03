import numpy as np
from matplotlib import pyplot as plt

def human_traj_generator(human_vel, dt):
    traj_list = []
    init= [0.,0.,0.]
    dist = human_vel* dt


    ################
    # straight trajectory
    # traj = [init]
    # for i in range(1, 20):
    #     x = traj[i-1][0] + dist 
    #     traj.append([x,0.,0.])

    # theta = 30 * np.pi/180
    # for i in range(i+1, i+6):
    #     x = traj[i-1][0] + dist * np.cos(theta)
    #     y = traj[i-1][1] + dist * np.sin(theta)
    #     traj.append([x,y,theta])

    # for i in range(i+1, i+7):
    #     x = traj[i-1][0] + dist 
    #     traj.append([x,y,0.])

    # traj_list.append(traj)


    ################
    # left/right turn
    turn = 6 * np.pi/180
    traj = [[-1.8,1.6,-1.46]]
    # traj= [init]
    for i in range(1, 22):
        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    turn = -theta 
    x = traj[i][0] + dist * np.cos(turn + traj[i][2])
    y = traj[i][1] + dist * np.sin(turn + traj[i][2])
    theta = turn + traj[i][2]
    traj.append([x,y,theta])

    turn = -7 * np.pi/180
    for i in range(i+2, i+10):
        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)



    ################
    # S shape
    # traj = [init]
    # traj = [[-2.08, -1.6, 1.22]]
    # fir = 15
    # sec = 5
    # thi = 15
    # for i in range(1,15+fir+sec+thi):
    #     if i <15+fir :
    #         turn = -5 * np.pi/180
    #     elif i == 15+fir:
    #         turn = -theta    
    #     elif i < 15+fir+sec :
    #         turn = 0 * np.pi/180
    #     elif i == 15+fir+sec:
    #         turn = 1.30   
    #     else:    
    #         turn = -5 * np.pi/180

    #     x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    #     y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    #     theta = turn + traj[i-1][2]
    #     traj.append([x,y,theta])

    # traj_list.append(traj)



    # x = [traj[i][0] for i in range(len(traj))]
    # y = [traj[i][1] for i in range(len(traj))]
    # plt.plot(x,y)
    # plt.axis('equal')
    # plt.show()

    return traj_list