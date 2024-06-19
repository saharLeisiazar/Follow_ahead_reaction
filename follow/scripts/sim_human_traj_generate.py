import numpy as np

def human_traj_generator(human_vel, dt):
    traj_list = []
    init= [-3.,0.,0.]
    dist = human_vel* dt

    ################
    # U shape
    traj = [init]
    for i in range(1,30):
        if i <5 or i>25:
            turn = 0
        else:
            turn = 8.5 * np.pi/180

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)

    ################
    # U shape
    traj = [init]
    for i in range(1,30):
        if i <5 or i>25:
            turn = 0
        else:
            turn = -8.5 * np.pi/180

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)

    ################
    # S shape
    traj = [init]
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

    traj_list.append(traj)

    ################
    # S shape
    traj = [init]
    for i in range(1,50):
        if i <10 :
            turn = -5 * np.pi/180
        elif i < 30:
            turn = 5 * np.pi/180
        else:    
            turn = -5 * np.pi/180

        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)


    ################
    # right turn
    turn = -4 * np.pi/180
    traj = [init]
    for i in range(1, 15):
        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)

    ################
    # left turn
    turn = 4 * np.pi/180
    traj = [init]
    for i in range(1, 15):
        x = traj[i-1][0] + dist * np.cos(turn + traj[i-1][2])
        y = traj[i-1][1] + dist * np.sin(turn + traj[i-1][2])
        theta = turn + traj[i-1][2]
        traj.append([x,y,theta])

    traj_list.append(traj)


    ################
    #straight trajectory
    traj = [init]
    for i in range(1, 10):
        x = traj[i-1][0] + dist 
        traj.append([x,0.,0.])

    traj_list.append(traj)


    return traj_list