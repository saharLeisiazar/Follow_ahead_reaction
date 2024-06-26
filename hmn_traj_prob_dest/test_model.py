import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


noise = 0.001

class LSTMModel2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel2D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.last(out)
        return out
    

##### Load model
path = '/home/sahar/catkin_ws/src/Follow_ahead_reaction/hmn_traj_prob_dest/'
dir = path+ 'desired_freq:5_tanh_power_value:0.2_seq_length:15_input_length:14_batch_size:64_hidden_size:64_num_epochs:10005_learning_rate:0.01_scheduler_step:5000'
model_name = dir + '/model.pth'
model = torch.load(model_name).cuda()
model.eval()


# Load trajectories and predict next point
seq_length = 15 
input_length = 14

# threshold = 50 * np.pi/180
directory = path+'trajectories_5hz'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        print(filename)
        file_path = os.path.join(directory, filename)
        data = np.genfromtxt(file_path, delimiter=',')

        for i in range(len(data) - seq_length):
            seq = np.array(data[i:i+seq_length])
            seq += np.random.uniform(-noise, noise, seq.shape)
            seq = torch.from_numpy(seq).float()

            x = seq - seq[input_length-1]
            x = x[:input_length].cuda()

            y_pred = model(torch.unsqueeze(x, 0)).detach().cpu()
            x = x.squeeze().detach().cpu()
            x += seq[input_length-1]
            y_pred = y_pred.squeeze()

            p1 = x[-1]
            p2 = x[-2]
            theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
            
            fig = plt.figure()
            ax=fig.add_subplot(111)    
            plt.plot(x[:, 0], x[:, 1], color='blue')
            y_pred = torch.flip(y_pred, [0]) # left, st, right   -> right, st, left

            dist_to_goal = 2.
            for j in range(y_pred.shape[0]):
                x_goal = dist_to_goal*np.cos(theta+ (1*(j-1))) + p1[0]
                y_goal = dist_to_goal*np.sin(theta+ (1*(j-1))) + p1[1]
                plt.scatter(x_goal, y_goal, color='red', s= 100)
                plt.text(x_goal, y_goal, str(round(y_pred[j].item(),2)), fontsize=12, color='black')
                    
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.xlim(-1., 6)
            if filename == 's.csv':
                ax.set_xlim([-2., 25])
                ax.set_ylim([-4., 8])
            else:
                ax.set_xlim([-2., 8])
                ax.set_ylim([-8., 8])
            ax.set_aspect('equal')
            # plt.ylim(-10, 10)
            plt.title('Ground Truth vs Predicted')
            # plt.axis('equal')
            # plt.legend()
            folder = filename[:-4]
            if not os.path.exists(dir + '/include/' + folder):
                os.makedirs(dir+ '/include/' + folder)

            fig.savefig( dir+ '/include/' + folder + '/' + str(i) + '.png')
            plt.close()




