import numpy as np
import torch
import matplotlib.pyplot as plt
from RL_interface import RL_model


file_name = 'a2c_len_50_random_walk_new_sahar_rew_small_net.zip'
model_directory = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/follow/include/' + file_name 
model = RL_model()
model.load_model(model_directory, policy='a2c')


X = np.linspace(1., 5., 100)
Y = np.linspace(-2., 3., 100)

R = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        state = np.array([[X[i], Y[j], 0.], [0., 0., 0.]])
        obs = np.concatenate([state[0,:2] - state[1,:2] , [state[1,2]], [state[0,2]]])
        obs = torch.FloatTensor(obs).unsqueeze(0)
        policy='a2c'
        R[i, j] = model.evaluate_state(obs, policy=policy)


plt.contourf(X, Y-0.8 , R.T, cmap='hot')
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap Plot')
plt.axis('equal')

plt.show()
print()