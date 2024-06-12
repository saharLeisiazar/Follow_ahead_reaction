from yacs.config import CfgNode
from typing import Dict, List, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from joblib import delayed, Parallel
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from torch.distributions.normal import Normal

from visualization.density_plot import plot_density
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import griddata, RBFInterpolator, interp2d
from scipy.stats import multivariate_normal


class Visualizer(ABC):
    def __init__(self, cfg: CfgNode):
        pass

    @abstractmethod
    def __call__(self, dict_list: List[Dict]) -> None:
        pass

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()


class TP_Visualizer(Visualizer):
    def __init__(self, cfg: CfgNode):
        self.model_name = cfg.MODEL.TYPE

        self.output_dir = Path(cfg.OUTPUT_DIR) / "visualize"
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = cfg.DATA.DATASET_NAME

        import dill
        env_path = Path(cfg.DATA.PATH) / cfg.DATA.TASK / \
            "processed_data" / f"{cfg.DATA.DATASET_NAME}_test.pkl"
        with open(env_path, 'rb') as f:
            self.env = dill.load(f, encoding='latin1')

        # TODO: params for other nodes
        from data.TP.trajectron_dataset import hypers
        node = 'PEDESTRIAN'
        self.state = hypers[cfg.DATA.TP.PRED_STATE][node]
        mean, std = self.env.get_standardize_params(self.state, node)
        self.mean = mean
        self.std = std
        self.std = self.env.attention_radius[('PEDESTRIAN', 'PEDESTRIAN')]

        self.num_grid = 100
        if hasattr(self.env, "gt_dist"):
            self.gt_dist = self.env.gt_dist
        else:
            self.gt_dist = None

        self.observe_length = cfg.DATA.OBSERVE_LENGTH

    ## Added: The categorize_predictions method categorizes the predicted next steps into left, straight, and right based on their angles relative to the current direction of movement.
    ## It calculates probabilities for each direction by counting the number of predicted points in each category and normalizing these counts.
    def categorize_predictions(self, obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """
        Categorize predictions into left, straight, and right based on the current direction.
        
        Args:
            obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
            pred (np.ndarray): (N_seqs, N_predictions, [x, y])
        
        Returns:
            np.ndarray: Categorized predictions and their probabilities (left, straight, right)
        """
        categories = np.zeros((obs.shape[0], 3))  # Probabilities for left, straight, right
        directions = np.array(['left', 'straight', 'right'])

        for i in range(obs.shape[0]):
            last_obs = obs[i, -1]
            second_last_obs = obs[i, -2]
            current_direction = last_obs - second_last_obs
            current_angle = np.arctan2(current_direction[1], current_direction[0])

            left_prob = 0
            straight_prob = 0
            right_prob = 0

            for j in range(pred.shape[1]):
                pred_direction = pred[i, j] - last_obs
                pred_angle = np.arctan2(pred_direction[1], pred_direction[0])
                angle_diff = np.degrees(pred_angle - current_angle)
                angle_diff = (angle_diff + 360) % 360  # Normalize angle to [0, 360)

                if 345 <= angle_diff or angle_diff < 15:
                    straight_prob += 1
                elif 15 <= angle_diff < 180:
                    right_prob += 1
                elif 180 <= angle_diff < 345:
                    left_prob += 1

            total = left_prob + straight_prob + right_prob
            if total > 0:
                categories[i, 0] = left_prob / total
                categories[i, 1] = straight_prob / total
                categories[i, 2] = right_prob / total

        return categories
    
    ## Added: The plot_direction_predictions method plots the observed trajectory, ground truth trajectory, and arrows indicating the predicted directions (left, straight, right) along with their probabilities.
    def plot_direction_predictions(self, obs: np.ndarray, gt: np.ndarray, pred: np.ndarray, index: Tuple, max_pos: np.ndarray, min_pos: np.ndarray, prob: np.ndarray) -> None:
        """
        Plot observed steps, ground truth trajectory, and the probability of three directions for the predicted step.
        
        Args:
            obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
            gt (np.ndarray): (N_seqs, N_timesteps, [x, y])
            pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x, y])
            index (Tuple): Identifier for the data
            max_pos (np.ndarray): Maximum positions for plot limits
            min_pos (np.ndarray): Minimum positions for plot limits
            prob (np.ndarray): Probability of three directions (left, straight, right)
        """

        N_seqs, N_timesteps, N_trials, N_dim = pred.shape
        gt_vis = np.zeros([N_seqs, N_timesteps+1, N_dim])
        gt_vis[:, 0] = obs[:, -1]
        gt_vis[:, 1:] = gt

        pred_vis = np.zeros([N_seqs, N_timesteps+1, N_trials, N_dim])
        pred_vis[:, 0] = obs[:, -1][:, None]
        pred_vis[:, 1:] = pred

        f, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        
        # Set static min and max positions        
        # ax.set_xlim(min_pos[0], max_pos[0])
        # ax.set_ylim(min_pos[1], max_pos[1])

        # Calculate dynamic min and max positions
        all_x = np.concatenate([obs[:, :, 0].flatten(), gt_vis[:, :, 0].flatten(), pred_vis[:, :, :, 0].flatten()])
        all_y = np.concatenate([obs[:, :, 1].flatten(), gt_vis[:, :, 1].flatten(), pred_vis[:, :, :, 1].flatten()])
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()
        
        ax.set_xlim(min_x-1, max_x+1)
        ax.set_ylim(min_y-1, max_y+1)

        # Plot observed trajectory
        for j in range(N_seqs):
            sns.lineplot(x=obs[j, :, 0], y=obs[j, :, 1], color='black', legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[j, :, 0], y=gt_vis[j, :, 1], color='blue', legend='brief', label="GT", marker='o')
            
            # Plot predicted directions with probabilities
            last_obs = obs[j, -1]
            second_last_obs = obs[j, -2]
            current_direction = last_obs - second_last_obs
            current_direction = current_direction / np.linalg.norm(current_direction)  # Normalize the vector

            # Define offsets
            directions = ['left', 'straight', 'right']
            colors = ['red', 'green', 'blue']
            offsets = [
                np.array([-current_direction[1], current_direction[0]]),  # Perpendicular to the left
                current_direction,  # Same direction as current
                np.array([current_direction[1], -current_direction[0]])  # Perpendicular to the right
            ]
            arrow_scale = 0.2  # Scale factor to shorten the arrows
            for i in range(3):
                direction = directions[i]
                color = colors[i]
                probability = prob[j, i]
                offset = offsets[i] * arrow_scale  # Scale down the offset

                next_step = last_obs + offset
                ax.arrow(last_obs[0], last_obs[1], next_step[0] - last_obs[0], next_step[1] - last_obs[1], head_width=0.05, head_length=0.1, fc=color, ec=color)
                ax.text(next_step[0], next_step[1], f'{probability:.2f}', color=color, fontsize=12, verticalalignment='bottom')
        img_path = self.output_dir / f"direction_predictions_{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}.png"
        plt.savefig(img_path)
        print("directon prediction saved to " + str(img_path))
        plt.close()


    def __call__(self, dict_list: List[Dict]) -> None:
        index = dict_list[0]['index']
        min_pos, max_pos = self.get_minmax(index)

        # (batch, timesteps, [x,y])
        obs = self.to_numpy(dict_list[0]['obs'][:, :, 0:2])
        gt = self.to_numpy(dict_list[0]['gt'])

        pred = []
        for d in dict_list:
            pred.append(self.to_numpy(d[("pred", 0)][:, :, None]))
            assert np.all(obs == self.to_numpy(d["obs"][:, :, 0:2]))
            assert np.all(gt == self.to_numpy(d["gt"]))

        # (batch, timesteps, num_trials, [x,y])
        pred = np.concatenate(pred, axis=2)
        for i in range(len(obs)):
            self.plot2d_trajectories(obs[i:i+1],
                                     gt[i:i+1],
                                     pred[i:i+1],
                                     index[i],
                                     max_pos,
                                     min_pos)

        if ("prob", 0) in dict_list[0]:
            path_density_map = self.output_dir / "density_map"
            path_density_map.mkdir(exist_ok=True)

            xx, yy = self.get_grid(index)

            for k in dict_list[0].keys():
                if k[0] == "prob":
                    update_step = k[1]
                    prob = dict_list[0][k]

                    bs, _, timesteps = prob.shape

                    obs = self.to_numpy(dict_list[0]['obs'])[..., :2]
                    gt = self.to_numpy(dict_list[0]['gt'])
                    traj = np.concatenate([obs, gt], axis=1)
                    obs = traj[:, :self.observe_length + update_step]
                    gt = traj[:, self.observe_length + update_step-1:]

                    zz_list = []
                    for j in range(timesteps):
                        zz = prob[0, :, j].reshape(xx.shape)
                        zz /= np.max(zz)
                        plot_density(xx, yy, zz, path=path_density_map / f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_{j}.png",
                                     traj=[obs[i], gt[i]])
                        zz_list.append(zz)

                    zz_sum = sum(zz_list)
                    plot_density(xx, yy, zz_sum,
                                 path=path_density_map /
                                 f"update{update_step}_{index[i][0]}_{index[i][1]}_{index[i][2].strip('PEDESTRIAN/')}_sum.png",
                                 traj=[obs[i], gt[i]])

    def prob_to_grid(self, dict_list: List[Dict]) -> List:
        if ("prob", 0) in dict_list[0]:
            index = dict_list[0]['index']
            min_pos, max_pos = self.get_minmax(index)
            xx, yy = self.get_grid(index)

            for data_dict in dict_list:
                data_dict["grid"] = [xx, yy]
                data_dict["minmax"] = [min_pos, max_pos]
                for k in list(data_dict.keys()):
                    if k[0] == "prob":
                        prob = data_dict[k]
                        if type(prob) == torch.Tensor:
                            prob = self.to_numpy(prob)
                            batch, _, timesteps, _ = prob.shape

                            zz_batch = []
                            for i in range(batch):
                                # zz_timesteps = Parallel(n_jobs=timesteps)(delayed(self.griddata_on_cluster)(i, prob, xx, yy, max_pos, min_pos, j)
                                #                                           for j in range(timesteps))
                                zz_timesteps = [self.griddata_on_cluster(i, prob, xx, yy, max_pos, min_pos, j) for j in range(timesteps)]
                                zz_timesteps = np.stack(
                                    zz_timesteps, axis=-1).reshape(-1, timesteps)
                                zz_batch.append(zz_timesteps)

                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch

                        elif self.model_name == "Trajectron" or self.model_name == "GT_Dist":
                            value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[
                                None, :, None].tile(1, 1, prob.mus.shape[2], 1).cuda()
                            zz_batch = torch.exp(prob.log_prob(value))
                            data_dict[k] = self.to_numpy(zz_batch)

                        else:
                            zz_batch = []
                            for i in range(len(prob)):
                                zz_timesteps = []
                                for kernel in prob[i]:
                                    zz = kernel(torch.Tensor(
                                        np.array([xx.flatten(), yy.flatten()]).T).cuda())
                                    zz_timesteps.append(zz.cpu().numpy())
                                zz_timesteps = np.stack(zz_timesteps, axis=1)
                                zz_batch.append(zz_timesteps)
                            zz_batch = np.stack(zz_batch, axis=0)
                            data_dict[k] = zz_batch

                        if ("gt_traj_log_prob", k[1]) not in data_dict:
                            gt = data_dict["gt"][:, k[1]:].cpu()
                            timesteps = gt.shape[1]
                            gt_traj_prob = np.array([interp2d(xx, yy, zz_batch[:, :, t])(
                                gt[:, t, 0], gt[:, t, 1]) for t in range(timesteps)])
                            gt_traj_log_prob = torch.log(
                                torch.Tensor(gt_traj_prob)).squeeze()
                            if torch.sum(torch.isnan(gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)) > 0:
                                mask = torch.isnan(
                                    gt_traj_log_prob) + torch.isinf(gt_traj_log_prob)
                                value = torch.min(gt_traj_log_prob[~mask])
                                gt_traj_log_prob = torch.nan_to_num(
                                    gt_traj_log_prob, nan=value, neginf=value)
                            data_dict[("gt_traj_log_prob", k[1])
                                      ] = gt_traj_log_prob[None]

                if self.gt_dist is not None:  # assume simfork
                    bs, timesteps, d = data_dict["gt"].shape
                    split = True
                    value = torch.Tensor(np.array([xx.flatten(), yy.flatten()])).transpose(0, 1)[
                        None, :, None].tile(1, 1, timesteps, 1).cuda()
                    if split:
                        gaussians = []
                        for i in range(len(self.gt_dist)):
                            gt_base_traj = torch.Tensor(
                                self.gt_dist[[i] * bs, -timesteps:]).cuda()
                            gaussians.append(
                                Normal(gt_base_traj[..., :2], gt_base_traj[..., 2:]))
                        data_dict["gt_prob"] = sum(
                            [torch.exp(g.log_prob(value).sum(dim=-1)) for g in gaussians])
                    else:
                        gt_base_traj = torch.Tensor(
                            self.gt_dist[data_dict["gt"][:, -1, 1] < 0, -timesteps:]).cuda()
                        gaussian = Normal(
                            gt_base_traj[:, :2], gt_base_traj[:, 2:])
                        data_dict["gt_prob"] = torch.exp(
                            gaussian.log_prob(value).sum(dim=-1))

        return dict_list

    ## Added
    def additional_plot(self, dict_list: List[Dict]) -> None:
        index = dict_list[0]['index']
        min_pos, max_pos = self.get_minmax(index)

        obs = self.to_numpy(dict_list[0]['obs'][:, :, 0:2])
        gt = self.to_numpy(dict_list[0]['gt'])

        pred = []
        for d in dict_list:
            pred.append(self.to_numpy(d[("pred", 0)][:, :, None]))
            assert np.all(obs == self.to_numpy(d["obs"][:, :, 0:2]))
            assert np.all(gt == self.to_numpy(d["gt"]))

        pred = np.concatenate(pred, axis=2)

        categorized_prob = self.categorize_predictions(obs, pred[:, 0, :, :])

        for i in range(len(obs)):
            self.plot_direction_predictions(obs[i:i+1], gt[i:i+1], pred[i:i+1], index[i], max_pos, min_pos, categorized_prob[i:i+1])

    def get_grid(self, index):
        min_pos, max_pos = self.get_minmax(index)
        xs = np.linspace(min_pos[0], max_pos[0], num=self.num_grid)
        ys = np.linspace(min_pos[1], max_pos[1], num=self.num_grid)
        xx, yy = np.meshgrid(xs, ys)

        return xx, yy

    def get_minmax(self, index):
        idx = [s.name for s in self.env.scenes].index(index[0][0])
        max_pos, min_pos = self.env.scenes[idx].calculate_pos_min_max()
        max_pos += 0.05 * (max_pos - min_pos)
        min_pos -= 0.05 * (max_pos - min_pos)
        return min_pos, max_pos

    def griddata_on_cluster(self, i, prob, xx, yy, max_pos, min_pos, j):
        prob_ = prob[i, :, j]
        prob_ = prob_[np.where(np.isinf(prob_).sum(axis=1) == 0)]
        prob_ = prob_[np.where(np.isnan(prob_).sum(axis=1) == 0)]
        lnk = linkage(prob_[:, :-1],
                      method='single',
                      metric='euclidean')
        idx_cls = fcluster(lnk, t=np.linalg.norm(max_pos-min_pos)*0.003,
                           criterion='distance')
        idx_cls -= 1

        zz_ = []
        for c in range(np.max(idx_cls)+1):
            try:
                zz_.append(griddata(prob_[idx_cls == c, :-1],
                                    prob_[idx_cls == c, -1],
                                    (xx, yy), method='linear',
                                    fill_value=0.0)
                           )
            except:
                pass

        zz = sum(zz_)

        intp = RBFInterpolator(np.array([xx, yy]).reshape(
            2, -1).T, zz.flatten(), smoothing=10, kernel='linear', neighbors=8)
        zz = intp(np.array([xx, yy]).reshape(2, -1).T).reshape(xx.shape)

        prob_ = prob[i, :, j, -1]
        prob_ = prob_[~np.isnan(prob_)]
        zz = np.clip(zz, a_min=0.0, a_max=np.percentile(prob_, 95))
        return zz

    def plot2d_trajectories(self,
                            obs:  np.ndarray,
                            gt:   np.ndarray,
                            pred: np.ndarray,
                            index: Tuple,
                            max_pos: np.ndarray,
                            min_pos: np.ndarray) -> None:
        """plot 2d trajectories

        Args:
            obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
            gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
            pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
            img_path (Path): Path
        """

        N_seqs, N_timesteps, N_trials, N_dim = pred.shape
        gt_vis = np.zeros([N_seqs, N_timesteps+1, N_dim])
        gt_vis[:, 0] = obs[:, -1]
        gt_vis[:, 1:] = gt

        pred_vis = np.zeros([N_seqs, N_timesteps+1, N_trials, N_dim])
        # (num_seqs, num_dim) -> (num_seqs, 1, num_dim)
        pred_vis[:, 0] = obs[:, -1][:, None]
        pred_vis[:, 1:] = pred

        f, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_pos[0], max_pos[0])
        ax.set_ylim(min_pos[1], max_pos[1])

        for j in range(N_seqs):
            sns.lineplot(x=obs[j, :, 0], y=obs[j, :, 1], color='black',
                         legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[j, :, 0], y=gt_vis[j, :, 1],
                         color='blue', legend='brief', label="GT", marker='o')
            for i in range(pred.shape[2]):
                if i == 0:
                    sns.lineplot(x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1],
                                 color='green', legend='brief', label="pred", marker='o')
                else:
                    sns.lineplot(
                        x=pred_vis[j, :, i, 0], y=pred_vis[j, :, i, 1], color='green', marker='o')

        img_path = self.output_dir / \
            f"{index[0]}_{index[1]}_{index[2].strip('PEDESTRIAN/')}.png"
        plt.savefig(img_path)
        print("saved to " + str(img_path))
        plt.close()


def plot2d_trajectories_samples(
        obs:  np.ndarray,
        gt:   np.ndarray,
        max_pos: np.ndarray,
        min_pos: np.ndarray) -> None:
    """plot 2d trajectories

    Args:
        obs (np.ndarray): (N_seqs, N_timesteps, [x, y])
        gt (np.ndarray): (N_seqs, N_timesteps, [x,y])
        pred (np.ndarray): (N_seqs, N_timesteps, N_trials, [x,y])
        img_path (Path): Path
    """

    N_seqs = len(gt)
    _, N_timesteps, N_dim = gt[0].shape

    f, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])

    for j in range(N_seqs):
        gt_vis = np.zeros([N_timesteps+1, N_dim])
        gt_vis[0] = obs[j][0, -1]
        gt_vis[1:] = gt[j][0]
        if j == 0:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                         legend='brief', label="obs", marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                         color='black', legend='brief', label="GT", marker='o')
        else:
            sns.lineplot(x=obs[j][0, :, 0], y=obs[j][0, :, 1], color='green',
                         marker='o')
            sns.lineplot(x=gt_vis[:, 0], y=gt_vis[:, 1],
                         color='black', marker='o')

    img_path = "gts.png"
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight')
    plt.close()
