#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import String
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import sys
sys.path.append("/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred/include/tutr")
from model import TrajectoryModel

sys.path.append("/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred/include/vae")
from social_vae import SocialVAE

class test_models():
    def __init__(self):
        rospy.init_node('test_modesl', anonymous=True)
        
        self.dir = "/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred"

        ####### TUTR model
        model_dir = '../include/tutr/best.pth'
        state_dic = torch.load(model_dir)
        self.tutr_model = TrajectoryModel(in_size=2, obs_len=8, pred_len=12, embed_size=128,
                                enc_num_layers=2, int_num_layers_list=[1,1], heads=4, forward_expansion=2).cuda()
        self.tutr_model.load_state_dict(state_dic)
        self.tutr_model = self.tutr_model.cuda()

        import pickle
        f = open(self.dir + '/include/tutr/univ_motion_modes.pkl', 'rb+')
        motion_modes = pickle.load(f)
        f.close()
        self.tutr_motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

        self.tutr_x = torch.zeros((1,2)).cuda()

        ####### VAE model
        self.vae_model = SocialVAE(horizon=12, ob_radius=8, hidden_dim=256)
        ckpt = self.dir + "/include/vae/ckpt-best"
        state_dict = torch.load(ckpt, map_location='cpu')
        self.vae_model.load_state_dict(state_dict["model"])
        # self.vae_model = self.vae_model.to('cuda:1')
        self.vae_x = torch.zeros((1,6))


        self.vis_x = np.zeros((1,2))
        self.threshold = 20*np.pi/180
        self.ind = 0
        rospy.Subscriber('topic', String, self.callbackFunction)



    def callbackFunction(self, data):
        with open(self.dir + "/include/data.txt", "r") as input_file:
            for line in input_file:
                columns = line.split()
                x = float(columns[2])
                y = float(columns[3])
                data = [x,y]

                tutr_probs = self.TUTR(data)
                vae_probs = self.VAE(data)  
                self.vis(data, tutr_probs, vae_probs)


    def vis(self, data, tutr_probs, vae_probs):
        data = np.array(data).reshape(1,-1)
        
        self.vis_x = np.append(self.vis_x, data, axis=0)
        dist_to_goal = 0.5

        if len(self.vis_x)>8:
            self.vis_x = self.vis_x[-8:]

            x = self.vis_x[:,0]
            y = self.vis_x[:,1]

            plt.figure()
            plt.plot(x, y, ls='-.', lw=2.0, color='blue')

            p1 = self.vis_x[-1, :2]
            p2 = self.vis_x[-2, :2]
            theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])

            x_goal = dist_to_goal*np.cos(theta+ (0.75)) + p1[0]
            y_goal = dist_to_goal*np.sin(theta+ (0.75)) + p1[1]
            plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
            plt.text(x_goal-1, y_goal, "tutr:"+str("%.2f" % tutr_probs["left"]),  ha='center')
            plt.text(x_goal-1, y_goal+0.3, "vae:"+str("%.2f" % vae_probs["left"]),  ha='center')

            x_goal = dist_to_goal*np.cos(theta- (0.75)) + p1[0]
            y_goal = dist_to_goal*np.sin(theta- (0.75)) + p1[1]
            plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
            plt.text(x_goal+1, y_goal, "tutr:"+str("%.2f" % tutr_probs["right"]),  ha='center')
            plt.text(x_goal+1, y_goal+0.3, "vae:"+str("%.2f" % vae_probs["right"]),  ha='center')

            x_goal = dist_to_goal*np.cos(theta) + p1[0]
            y_goal = dist_to_goal*np.sin(theta) + p1[1]
            plt.scatter(x_goal, y_goal, marker='*', color='red', s=300)
            plt.text(x_goal, y_goal+1, "tutr:"+str("%.2f" % tutr_probs["straight"]),  ha='center')
            plt.text(x_goal, y_goal+1+0.3, "vae:"+str("%.2f" % vae_probs["straight"]),  ha='center')

            # plt.tight_layout()
            plt.axis('scaled')
            plt.xlim([-1,6])
            plt.ylim([1,9])
            save_path = "/home/sahar/catkin_ws/src/Follow_ahead_reaction/human_pred/fig/"
            plt.savefig(save_path + str(self.ind) + '.png')
            self.ind +=1

        return


    def TUTR(self, data):
        self.tutr_model.eval()
        new = torch.unsqueeze(torch.tensor(data),0).cuda()
        self.tutr_x = torch.cat((self.tutr_x,new),dim=0)

        left = 0.
        right = 0.
        straight = 0.

        if self.tutr_x.shape[0] > 8:
            self.tutr_x = self.tutr_x[-8:]


            if True:  #self.translation:
                origin = self.tutr_x[-1] # [1, 2]
                self.tutr_x = self.tutr_x - origin
            
            if True:   #self.rotation:
                ref_point = self.tutr_x[0]
                angle = torch.atan2(ref_point[1], ref_point[0])
                rot_mat = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                          [torch.sin(angle), torch.cos(angle)]]).cuda()
                self.tutr_x = torch.matmul(self.tutr_x, rot_mat)


            ped_obs = torch.unsqueeze(self.tutr_x,0)
            neis_obs = torch.unsqueeze(ped_obs, 0)
            mask = torch.ones((1,1,1)).cuda()
            pred_trajs, scores = self.tutr_model(ped_obs, neis_obs, self.tutr_motion_modes, mask, None, test=True)
            top_k_scores = torch.topk(scores, k=20, dim=-1).values
            top_k_scores = F.softmax(top_k_scores, dim=-1)
            pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], -1, 2)
            pred_trajs = torch.squeeze(pred_trajs)

            #### human's current orientation
            ped_obs = torch.squeeze(ped_obs)
            top_k_scores = top_k_scores.detach().tolist()

            p1 = ped_obs[-1, :2]
            p2 = ped_obs[-2, :2]
            theta = torch.atan2(p1[1]-p2[1], p1[0]-p2[0])

            goals = pred_trajs[:, -1, :]
            dp = goals - p1
            beta = torch.atan2(dp[:,1], dp[:,0])
            alpha = beta - theta

            for a,p in zip(alpha, top_k_scores):
                if a > self.threshold:
                    left+=p
                elif a < -self.threshold:
                    right+=p
                else:
                    straight+=p

            print("############ Prediction for TURT")
            print("left: ", left)   
            print("right: ", right) 
            print("straight: ", straight) 


        return {'left': left, 'right': right, 'straight': straight}
    
    def VAE(self, data):
        self.vae_model.eval()
        fpc = 32
        
        left = 0.
        right = 0.
        straight = 1e-9
        
        neighbor = torch.ones((20,1,1,6)) * 1e9
        x = data[0]
        y = data[1]
        if len(self.vae_x):
            vx = x - self.vae_x[-1][0]
            vy = y - self.vae_x[-1][1]
            ax = vx - self.vae_x[-1][2]
            ay = vy - self.vae_x[-1][3]

        new = torch.unsqueeze(torch.tensor([x, y, vx, vy, ax, ay]),0)
        self.vae_x = torch.cat((self.vae_x,new),dim=0)

        if self.vae_x.shape[0] > 8:
            self.vae_x = self.vae_x[-8:]
            self.vae_x = torch.unsqueeze(self.vae_x, 1)
        
            # disable fpc testing during training
            y_ = []
            for _ in range(fpc):
                y_.append(self.vae_model(self.vae_x, neighbor, n_predictions=20, y=None))
            y_ = torch.cat(y_, 0)

            #### human's current orientation
            self.vae_x = torch.squeeze(self.vae_x)
            y_ = torch.squeeze(y_)
            p1 = self.vae_x[-1, :2]
            p2 = self.vae_x[-2, :2]
            theta = torch.atan2(p1[1]-p2[1], p1[0]-p2[0])

            goals = y_[:, -1, :]
            dp = goals - p1
            beta = torch.atan2(dp[:,1], dp[:,0])
            alpha = beta - theta

            for a in alpha:
                if a > self.threshold:
                    left+=1
                elif a < -self.threshold:
                    right+=1
                else:
                    straight+=1

            # print("############## VAE prediction")
            # print("left: ", left/(left+right+straight))   
            # print("right: ", right/(left+right+straight)) 
            # print("straight: ", straight/(left+right+straight)) 

        return {'left': left/(left+right+straight), 'right': right/(left+right+straight), 'straight': straight/(left+right+straight)}



if __name__ == '__main__':
    test_models()
    rospy.spin()