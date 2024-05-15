import torch
torch.cuda.empty_cache()

import time

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred/include/vae")
from social_vae import SocialVAE

dir = "/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred"
threshold = 20 *np.pi/180

####### VAE model
vae_model = SocialVAE(horizon=12, ob_radius=8, hidden_dim=256)
ckpt = dir + "/include/vae/ckpt-best"
state_dict = torch.load(ckpt, map_location='cpu')
vae_model.load_state_dict(state_dict["model"])
# vae_model = vae_model.to("cuda:0")
vae_x = torch.zeros((1,6))#.to("cuda:0")
vis_x = np.zeros((1,2))
ind = 0

def VAE(data):
    vae_model.eval()
    fpc = 3
    
    left = 0.
    right = 0.
    straight = 1e-9
    
    neighbor = torch.ones((20,1,1,6)) * 1e9
    x = data[0]
    y = data[1]
    global vae_x
    if len(vae_x):
        vx = x - vae_x[-1][0]
        vy = y - vae_x[-1][1]
        ax = vx - vae_x[-1][2]
        ay = vy - vae_x[-1][3]

    new = torch.unsqueeze(torch.tensor([x, y, vx, vy, ax, ay]),0)
    vae_x = torch.cat((vae_x,new),dim=0)

    if vae_x.shape[0] > 8:
        vae_x = vae_x[-8:]
        vae_x = torch.unsqueeze(vae_x, 1)
    
        # disable fpc testing during training
        y_ = []
        for _ in range(fpc):
            # print()
            # t = time.time()
            y_.append(vae_model(vae_x, neighbor, n_predictions=20, y=None))
            # print(time.time()-t)
        y_ = torch.cat(y_, 0)

        #### human's current orientation
        vae_x = torch.squeeze(vae_x)
        y_ = torch.squeeze(y_)
        p1 = vae_x[-1, :2]
        p2 = vae_x[-2, :2]
        theta = torch.atan2(p1[1]-p2[1], p1[0]-p2[0])

        goals = y_[:, -1, :]
        dp = goals - p1
        beta = torch.atan2(dp[:,1], dp[:,0])
        alpha = beta - theta

        for a in alpha:
            if a > threshold:
                left+=1
            elif a < -threshold:
                right+=1
            else:
                straight+=1

        # print("############## VAE prediction")
        # print("left: ", left/(left+right+straight))   
        # print("right: ", right/(left+right+straight)) 
        # print("straight: ", straight/(left+right+straight)) 

    return {'left': left/(left+right+straight), 'right': right/(left+right+straight), 'straight': straight/(left+right+straight)}



def vis(data, tutr_probs = None, vae_probs= None):
    data = np.array(data).reshape(1,-1)
    
    global vis_x 
    vis_x = np.append(vis_x, data, axis=0)
    dist_to_goal = 0.5

    if len(vis_x)>8:
        vis_x = vis_x[-8:]

        x = vis_x[:,0]
        y = vis_x[:,1]

        plt.figure()
        plt.plot(x, y, ls='-.', lw=2.0, color='blue')

        p1 = vis_x[-1, :2]
        p2 = vis_x[-2, :2]
        theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])

        x_goal = dist_to_goal*np.cos(theta+ (0.75)) + p1[0]
        y_goal = dist_to_goal*np.sin(theta+ (0.75)) + p1[1]
        plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
        if tutr_probs:
            plt.text(x_goal-1, y_goal, "tutr:"+str("%.2f" % tutr_probs["left"]),  ha='center')
        if vae_probs:
            plt.text(x_goal-1, y_goal+0.3, "vae:"+str("%.2f" % vae_probs["left"]),  ha='center')

        x_goal = dist_to_goal*np.cos(theta- (0.75)) + p1[0]
        y_goal = dist_to_goal*np.sin(theta- (0.75)) + p1[1]
        plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
        if tutr_probs:
            plt.text(x_goal+1, y_goal, "tutr:"+str("%.2f" % tutr_probs["right"]),  ha='center')
        if vae_probs:
            plt.text(x_goal+1, y_goal+0.3, "vae:"+str("%.2f" % vae_probs["right"]),  ha='center')

        x_goal = dist_to_goal*np.cos(theta) + p1[0]
        y_goal = dist_to_goal*np.sin(theta) + p1[1]
        plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
        if tutr_probs:
            plt.text(x_goal, y_goal+1, "tutr:"+str("%.2f" % tutr_probs["straight"]),  ha='center')
        if vae_probs:
            plt.text(x_goal, y_goal+1+0.3, "vae:"+str("%.2f" % vae_probs["straight"]),  ha='center')

        # plt.tight_layout()
        plt.axis('scaled')
        plt.xlim([-1,6])
        plt.ylim([1,9])
        save_path = "/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred/fig/"
        global ind
        plt.savefig(save_path + str(ind) + '.png')
        ind +=1

    return



with open("/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred/include/u.txt", "r") as input_file:
    for line in input_file:
        columns = line.split()
        x = float(columns[2])
        y = float(columns[3])
        data = [x,y]

        # tutr_probs = self.TUTR(data)
        vae_probs = VAE(data)  
        vis(data, vae_probs = vae_probs)



print("Done")
