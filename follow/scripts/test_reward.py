import numpy as np
import torch
import matplotlib.pyplot as plt


def calculate_reward(state):
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
    # r_o += 10

    r = r_d + r_o 
    return r



X = np.linspace(-2., 4., 100)
Y = np.linspace(-3., 3., 100)

R = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        state = np.array([[X[i], Y[j], 0.], [0., 0., 0.]])
        R[i, j] = calculate_reward(state)


plt.contourf(X, Y , R.T, cmap='hot')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap Plot')
plt.axis('equal')

# fig.savefig('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/'+  file_name[:-4]   +'.png')
plt.show()
print()