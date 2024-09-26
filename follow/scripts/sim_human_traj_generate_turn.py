import numpy as np
import matplotlib.pyplot as plt


def human_traj_generator(human_vel, dt):
    traj_list = []
    init= [0.,0.,0.]
    dist = human_vel* dt
    length = 35
    ################
    # first turn
    traj = [init]
    for i in range(1,length):
        if i <25 :
            turn = 0
        elif i == 25:
            turn = 90 * np.pi/180    

        else:    
            turn = 0

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)

    x = []
    y = []
    for j in range(len(traj)):
        x.append(traj[j][0])
        y.append(traj[j][1])

    # plt.plot(x, y, 'ro')
    # plt.axis('equal')
    # plt.show()

   ################
    # second turn
    traj = [init]
    for i in range(1,length):
        idx = 3
        if i <25 - idx :
            turn = 0
        elif i < 25:
            turn = 90./idx * np.pi/180

        else:    
            turn = 0

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)
    
    x = []
    y = []
    for j in range(len(traj)):
        x.append(traj[j][0])
        y.append(traj[j][1])

    # plt.plot(x, y, 'ro')
    # plt.axis('equal')
    # plt.show()
    
    ################
    # third turn
    traj = [init]
    for i in range(1,length):
        idx = 6
        if i <25 - idx :
            turn = 0
        elif i < 25:
            turn = 90./idx * np.pi/180

        else:    
            turn = 0

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)
    
    x = []
    y = []
    for j in range(len(traj)):
        x.append(traj[j][0])
        y.append(traj[j][1])

    # plt.plot(x, y, 'ro')
    # plt.axis('equal')
    # plt.show()

    ################
    # forth turn
    traj = [init]
    for i in range(1,length):
        idx = 9
        if i <25 - idx :
            turn = 0
        elif i < 25:
            turn = 90./idx * np.pi/180

        else:    
            turn = 0

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)
    
    x = []
    y = []
    for j in range(len(traj)):
        x.append(traj[j][0])
        y.append(traj[j][1])

    # plt.plot(x, y, 'ro')
    # plt.axis('equal')
    # plt.show()

 



    ################
    # S shape
    # traj = [init]
    # for i in range(1,50):
    #     if i <10 :
    #         turn = -5 * np.pi/180
    #     elif i < 30:
    #         turn = 5 * np.pi/180
    #     else:    
    #         turn = -5 * np.pi/180

    #     x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    #     y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    #     theta = turn + traj[i-1][2]
    #     traj.append([x,y,theta])

    # traj_list.append(traj)


    ################
    # right turn
    # turn = -4 * np.pi/180
    # traj = [[0,0,1.]]
    # for i in range(1, 30):
    #     x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    #     y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    #     theta = turn + traj[i-1][2]
    #     traj.append([x,y,theta])

    # traj_list.append(traj)

    ################
    # left turn
    # turn = 4 * np.pi/180
    # traj = [init]
    # for i in range(1, 30):
    #     x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
    #     y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
    #     theta = turn + traj[i-1][2]
    #     traj.append([x,y,theta])

    # traj_list.append(traj)


    ################



    return traj_list