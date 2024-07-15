import numpy as np
import torch
import matplotlib.pyplot as plt
from RL_interface import RL_model


file_name = 'multiply_rewards_1.zip'
model_directory = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/' + file_name 
model = RL_model()
model.load_model(model_directory, policy='a2c')


X = np.linspace(-2., 4., 100)
Y = np.linspace(-3., 3., 100)

R = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        state = np.array([[X[i], Y[j], 0.], [0., 0., 0.]])
        obs = np.concatenate([state[0,:2] - state[1,:2] , [state[0,2]-state[1,2]]])
        obs = torch.FloatTensor(obs).unsqueeze(0)
        policy='a2c'
        R[i, j] = model.evaluate_state(obs, policy=policy)


fig = plt.contourf(X, Y , R.T, cmap='hot')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap Plot')
plt.axis('equal')

# fig.savefig('/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/'+  file_name[:-4]   +'.png')
plt.show()
print()