import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("./human_pred/include/vae")
sys.path.append("./human_pred/include/tutr")
from social_vae import SocialVAE
from model import TrajectoryModel  
from dataset import TrajectoryDataset

def main():

    dir = "./human_pred"
    threshold = 20 *np.pi/180

    ####### VAE model
    vae_model = SocialVAE(horizon=12, ob_radius=8, hidden_dim=256)
    ckpt = dir + "/include/vae/ckpt-best"
    state_dict = torch.load(ckpt, map_location='cpu')
    vae_model.load_state_dict(state_dict["model"])
    # vae_model = vae_model.to("cuda:0")
    global vae_x
    vae_x = torch.zeros((1,6))#.to("cuda:0")
    global vis_x 
    vis_x = np.zeros((1,2))
    global ind
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

    ####### TUTR model
    tutr_model = TrajectoryModel(
        in_size=2,
        obs_len=8,
        pred_len=12,
        embed_size=128,
        enc_num_layers=2,
        int_num_layers_list=[1, 1],
        heads=4,
        forward_expansion=2
    )

    # tutr_model = tutr_model.cuda()
    # tutr_model.load_state_dict(torch.load(dir + "/include/tutr/best_sfu_2.pth"))

    # test_dataset = TrajectoryDataset(dataset_path=dir + '/include/tutr/dataset/', dataset_name='u',
    #                                     dataset_type='test', translation=True, rotation=True, 
    #                                     scaling=False, obs_len=8)
    # test_loader = DataLoader(test_dataset, collate_fn=test_dataset.coll_fn, batch_size=128, shuffle=True, num_workers=8)

    # import pickle
    # f = open(dir + '/include/tutr/dataset/sfu_motion_modes.pkl', 'rb+')
    # motion_modes = pickle.load(f)
    # f.close()
    # motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()

    ckpt_tutr = dir + "/include/tutr/best_sfu_2.pth"
    state_dict_tutr = torch.load(ckpt_tutr, map_location='cpu')
    # # Print out the keys and shapes in the state_dict
    # for k, v in state_dict_tutr.items():
    #     print(f"{k}: {type(v)}")

    # Extract the model state_dict from the checkpoint
    model_state_dict = state_dict_tutr.get("model", state_dict)

    # Load the state_dict into the model
    tutr_model.load_state_dict(model_state_dict, strict=False)



    global tutr_x
    tutr_x = []

    # def TUTR(model, test_dataloader, motion_modes):
    #     model.eval()

    #     ade = 0
    #     fde = 0
    #     num_traj = 0
    #     threshold = 20 * np.pi / 180
    #     left = 0.0
    #     right = 0.0
    #     straight = 1e-9

    #     for (ped, neis, mask) in test_dataloader:
    #         ped = ped.cuda()
    #         neis = neis.cuda()
    #         mask = mask.cuda()

    #         ped_obs = ped[:, :8]  # Assuming args.obs_len = 8
    #         gt = ped[:, 8:]       # Assuming args.obs_len = 8
    #         neis_obs = neis[:, :, :8]

    #         with torch.no_grad():
    #             num_traj += ped_obs.shape[0]
    #             pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
    #             top_k_scores = torch.topk(scores, k=min(20, scores.shape[-1]), dim=-1).values
    #             top_k_scores = F.softmax(top_k_scores, dim=-1)
    #             pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
    #             gt_ = gt.unsqueeze(1)
    #             norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
    #             ade_ = torch.mean(norm_, dim=-1)
    #             fde_ = norm_[:, :, -1]
    #             min_ade, min_ade_index = torch.min(ade_, dim=-1)
    #             min_fde, min_fde_index = torch.min(fde_, dim=-1)

    #             print("ped_obs:", ped_obs)
    #             print("pred_trajs:", pred_trajs)
    #             # print("gt_:", gt_)

    #             for i in range(ped_obs.shape[0]):
    #                 p1 = ped_obs[i, -1, :]  # Last observed point
    #                 p2 = ped_obs[i, -2, :]  # Second to last observed point
    #                 theta = torch.atan2(p1[1] - p2[1], p1[0] - p2[0])
    #                 print('p1:', p1)
    #                 print('p2:', p2)

    #                 goals = pred_trajs[:, :, -1, :].detach().cpu().numpy().reshape(-1, 2)
    #                 # print('goals:', goals)
    #                 dp = goals - p1.cpu().numpy()
    #                 beta = np.arctan2(dp[:, 1], dp[:, 0])
    #                 alpha = beta - theta.item()

    #                 for a in alpha:
    #                     if a > threshold:
    #                         left += 1
    #                     elif a < -threshold:
    #                         right += 1
    #                     else:
    #                         straight += 1

    #     total = left + right + straight
    #     print(f"Directions: Left: {left}, Right: {right}, Straight: {straight}")

    #     return {'left': left / total, 'right': right / total, 'straight': straight / total}

    def TUTR(data):
        global tutr_x
        tutr_model.eval()
        num_k = 20

        left = 0.0
        right = 0.0
        straight = 1e-9

        obs_len = 8  
        pred_len = 12  
        num_modes = 20 

        tutr_x.append(data)

        if len(tutr_x) > 8:
            obs_data = torch.tensor(tutr_x[-8:]).unsqueeze(0)  # Convert last 8 points into a tensor [1, 8, 2]
            ped_obs = torch.tensor(obs_data).reshape(1, obs_len, 2)  # [1, obs_len, 2]
            
            neis_obs = torch.ones((1, 20, 8, 2)) * 1e9  # [1, 20, obs_len, 2] with dummy data

            motion_modes = torch.ones((20, pred_len, 2)) * 1e9  # [20, pred_len, 2] with dummy data

            mask = torch.ones((1, 20, 20)) * 1e9  # [1, 20, 20] with dummy data
            # print("mask shape:", mask.shape)  

            closest_mode_indices = torch.tensor([0])  # [1] with dummy data
            # print("closest_mode_indices shape:", closest_mode_indices.shape) 

            # Ensure correct dimensions for ped_seq
            ped_obs_expanded = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1)  # [1, 20, obs_len, 2]
            
            motion_modes_expanded = motion_modes.unsqueeze(0).repeat(ped_obs_expanded.shape[0], 1, 1, 1)  # [1, 20, pred_len, 2]
            # print("motion_modes_expanded shape:", motion_modes_expanded.shape)  

            ped_seq = torch.cat((ped_obs_expanded, motion_modes_expanded), dim=-2)  # [1, 20, seq_len, 2]
            ped_seq = ped_seq.reshape(ped_seq.shape[0], num_modes, -1)  # [1*20, seq_len*2]
            # print("ped_seq shape:", ped_seq.shape)
            # print("embedding weight shape:", tutr_model.embedding.weight.shape)
            # Expand ped_seq to the required shape [1, 20, 128]
            if ped_seq.shape[-1] < 128:
                pad_size = 128 - ped_seq.shape[-1]
                ped_seq = torch.nn.functional.pad(ped_seq, (0, pad_size))
            else:
                ped_seq = ped_seq[:, :, :128]

            # print("padded ped_seq shape:", ped_seq.shape)  # 打印形状以验证
            # print("ped_seq:", ped_seq)  # Log ped_seq

            # ped_embedding = tutr_model.embedding(ped_seq)  # [B*K, embed_size]
            # ped_embedding = ped_embedding.reshape(1, motion_modes.shape[0], -1)  # [1, K, embed_size]
            # print("ped_embedding shape:", ped_embedding.shape) 

            # Correct reshaping for input to mode_encoder
            ped_feat = tutr_model.mode_encoder(ped_seq) # [1, K, embed_size]
            scores = tutr_model.cls_head(ped_feat).squeeze()  # [1, K]
            print("ped_feat:", ped_feat)  
            # print("scores shape:", scores.shape)  
            print("Scores:", scores)  # Log scores

            # Top K selection and interaction
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices  # [1, num_k]
            top_k_indices = top_k_indices.flatten()  # [num_k]
            top_k_feat = ped_feat[:, top_k_indices, :]  # [1, num_k, embed_size]
            # print("top_k_feat shape:", top_k_feat.shape) 
            # print("Top K Indices:", top_k_indices.numpy())  # Log top K indices
            # print("Top K Features:", top_k_feat.detach().numpy())

            int_feats = tutr_model.spatial_interaction(top_k_feat, neis_obs, mask)  # [1, num_k, embed_size]
            pred_trajs = tutr_model.reg_head(int_feats).reshape(1, num_k, -1, 2)  # [1, num_k, pred_len, 2]
            # print("int_feats shape:", int_feats.shape)  
            # print("pred_trajs shape:", pred_trajs.shape)  
            # print("Interaction Features:", int_feats.detach().numpy())

            p1 = ped_obs[0, -1, :]
            p2 = ped_obs[0, -2, :]
            # print("p1 shape:", p1.shape)
            # print("p2 shape:", p2.shape)
            # print("p1:", p1)
            # print("p2:", p2)
            theta = torch.atan2(p1[1] - p2[1], p1[0] - p2[0])

            goals = pred_trajs[:, :, -1, :].detach().numpy().reshape(-1, 2)
            # print("Goals:", goals)
            dp = goals - p1.numpy()
            beta = np.arctan2(dp[:, 1], dp[:, 0])
            alpha = beta - theta.item()
            # print("Theta (current heading):", theta.item())
            # print("Beta (goal direction):", beta)
            # print("Alpha (angle difference):", alpha)
            # print("Theta:", theta.item(), "Alphas:", alpha)  # Debug output

            for a in alpha:
                if a > threshold:
                    left += 1
                elif a < -threshold:
                    right += 1
                else:
                    straight += 1
            total = left + right + straight
            print(f"Directions: Left: {left}, Right: {right}, Straight: {straight}")  # More debug output

            return {'left': left / total, 'right': right / total, 'straight': straight / total}
        else:
            print("Not enough data points collected yet.")
            return {'left': 0, 'right': 0, 'straight': 1}

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
            save_path = "./human_pred/fig/"
            global ind
            plt.savefig(save_path + str(ind) + '.png')
            ind +=1

        return



    with open("./human_pred/include/u.txt", "r") as input_file:
        for line in input_file:
            columns = line.split()
            x = float(columns[2])
            y = float(columns[3])
            data = [x,y]
            # print("data: ",data)

            tutr_probs = TUTR(data)
            vae_probs = VAE(data)  
            vis(data, vae_probs = vae_probs, tutr_probs = tutr_probs)



    print("Done")

if __name__ == '__main__':
    main()