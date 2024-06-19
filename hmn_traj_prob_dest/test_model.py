import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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
model_name = 'model.pth'
model = torch.load(model_name).cuda()
model.eval()


# Load trajectories and predict next point
seq_length = 6 
input_length = 5

# threshold = 50 * np.pi/180
directory = '../include/trajectories'
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        print(filename)
        file_path = os.path.join(directory, filename)
        data = np.genfromtxt(file_path, delimiter=',')

        for i in range(len(data) - seq_length):
            if i==20:
                print()
            seq = data[i:i+seq_length]
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
            if not os.path.exists('../include/' + folder):
                os.makedirs('../include/' + folder)

            fig.savefig( '../include/' + folder + '/' + str(i) + '.png')
            plt.close()



















            # fig = plt.figure()
            # x = np.squeeze(x, axis=0)
            # mu = np.squeeze(mu, axis=0)
            # pi = np.squeeze(pi, axis=0)


            # #### human's current orientation
            # p1 = x[-1, :2]
            # p2 = x[-2, :2]
            # theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])

            # dp = mu - p1
            # beta = np.arctan2(dp[:,1], dp[:,0])
            # alpha = beta - theta

            # left = 0.
            # right = 0.
            # straight = 0.
            # for a, p in zip(alpha, pi):
            #     if a > threshold:
            #         left+=p
            #     elif a < -threshold:
            #         right+=p
            #     else:
            #         straight+=p


            # plt.plot(x[:, 0], x[:, 1], 'bo', label='X_test')
            # plt.plot(y[0], y[1], 'go', label='GT')
            
            # dist_to_goal = 0.5
            # left = round(left, 2)
            # right = round(right, 2)
            # straight = round(straight, 2)

            # x_goal = dist_to_goal*np.cos(theta+ (0.5)) + p1[0]
            # y_goal = dist_to_goal*np.sin(theta+ (0.5)) + p1[1]
            # plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
            # plt.text(x_goal, y_goal, str(left),  ha='center')

            # x_goal = dist_to_goal*np.cos(theta- (0.5)) + p1[0]
            # y_goal = dist_to_goal*np.sin(theta- (0.5)) + p1[1]
            # plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
            # plt.text(x_goal, y_goal, str(right),  ha='center')

            # x_goal = dist_to_goal*np.cos(theta) + p1[0]
            # y_goal = dist_to_goal*np.sin(theta) + p1[1]
            # plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
            # plt.text(x_goal, y_goal, str(straight),  ha='center')

            # plt.legend()
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Test Data and Predicted Values')
            # plt.axis('equal')
            # # plt.xlim(-0.5, 7)

            # folder = filename[:-4]
            # fig.savefig( '../include/' + folder + '/' + str(i) + '.png')

