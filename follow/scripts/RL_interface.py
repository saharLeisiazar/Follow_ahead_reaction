from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from nav_env import Environment
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
from matplotlib.transforms import Affine2D
import torch
import numpy as np
import random
from stable_baselines3 import A2C
from stable_baselines3 import DDPG

DEVICE = 'cuda'


class RL_model:
    def __init__(self):
        return

    def load_model(self, model_path='', policy='a2c', env=Environment()):
        if policy == 'dqn':
            self.model = DQN('MlpPolicy', env, verbose=1, buffer_size=10000, learning_rate=1e-3, batch_size=32, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.02)
            self.model.q_net.load_state_dict(torch.load(model_path))
        elif policy == 'a2c':
            self.model = A2C.load(model_path)
        elif policy == 'ddpg':
            self.model = DDPG.load(model_path)
        else:
            raise Exception
        return self.model    

    def evaluate_state(self, state, action=None, policy='a2c'):
        assert action is None or policy == 'dqn'
        assert policy is not None
        state = state.to(DEVICE)
        if policy == 'dqn':
            q_values = self.model.policy.q_net(state).detach()
            q_values = q_values.flatten()
            if action is None:
                return torch.max(q_values)
            return q_values[action]
        elif policy == 'a2c':
            value = self.model.policy.predict_values(state) 
            return value
        elif policy == 'ddpg':
            q_values = self.model.critic(state).detach()
            q_values = q_values.flatten()
            if action is None:
                return torch.max(q_values)
            return q_values[action]
        else:
            raise Exception()


def convert_action(env:Environment, action):
    tmp = [0, np.pi/4, np.pi/2, np.pi * 3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4]
        
    agent_orientation = tmp[action % 8]

    if action < 8:
        agent_speed = env.agent_step_size1
    else: 
        agent_speed = env.agent_step_size2
    
    return agent_speed, agent_orientation

def evaluate_state(model, state, action=None, policy='a2c'):
    assert action is None or policy == 'dqn'
    assert policy is not None
    state = state.to(DEVICE)
    if policy == 'dqn':
        q_values = model.policy.q_net(state).detach()
        q_values = q_values.flatten()
        if action is None:
            return torch.max(q_values)
        return q_values[action]
    elif policy == 'a2c':
        value = model.policy.predict_values(state) 
        return value
    elif policy == 'ddpg':
        q_values = model.critic(state).detach()
        q_values = q_values.flatten()
        if action is None:
            return torch.max(q_values)
        return q_values[action]
    else:
        raise Exception()

def select_action(model, state):
    state = state.to(DEVICE)
    q_values = model.policy.q_net(state).detach()
    q_values = q_values.flatten()
    return torch.argmax(q_values).item()


def load_model(model_path='', policy='a2c', env=Environment()):
    if policy == 'dqn':
        model = DQN('MlpPolicy', env, verbose=1, buffer_size=10000, learning_rate=1e-3, batch_size=32, gamma=0.99, exploration_fraction=0.1, exploration_final_eps=0.02)
        model.q_net.load_state_dict(torch.load(model_path))
    elif policy == 'a2c':
        model = A2C.load(model_path)
    elif policy == 'ddpg':
        model = DDPG.load(model_path)
    else:
        raise Exception
    return model

def evaluate_agent(env, model, n_episodes=1, device='cpu', name='evaluate_agent', policy='dqn'):
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        return_ = 0
        while not done:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device='cpu')
                if policy=='a2c':
                    action, _states = model.predict(state, deterministic=True)
                    #breakpoint()
                    action = action.item()
                    value = model.policy.predict_values(state.to(DEVICE))  # This fetches the value estimate
                elif policy == 'dqn':
                    action = select_action(model=model, state=state)
                else:
                    raise Exception
                #action = torch.argmax(action_prob).item()
                state, reward, done, _ = env.step(action)
                return_ = return_ * 0.99 + reward
        print(f'##Return##: {return_}')

        # Plotting
        human_positions = np.array(env.history['human'])
        agent_positions = np.array(env.history['agent'])
        plt.figure(figsize=(8, 8))

        num_points = len(human_positions)
        color_map = np.array([
            np.linspace(0, 1, num_points),  # Red channel increases
            np.zeros(num_points),           # Green channel is zero
            np.linspace(1, 0, num_points)   # Blue channel decreases
        ]).T
        colors1 = [mcolors.to_rgba(c) for c in color_map]

        for i in range(num_points):
            plt.scatter(human_positions[i, 0], human_positions[i, 1], color=colors1[i], marker='o', label='Human' if i == 0 else "")
            plt.scatter(agent_positions[i, 0], agent_positions[i, 1], color=colors1[i], marker='x', label='Agent' if i == 0 else "")

        # Add labels and legends
        plt.legend()
        plt.title(f'Episode {episode+1} Trajectory')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.grid(True)
        plt.savefig(f'./{name}.png')
        plt.close()

def plot_evaluate_states(env, model, device='cpu', name='evaluate_states', policy='dqn'):
    fig, ax = plt.subplots()

    hum_pos = np.random.uniform(0, env.max_x, 2) #np.array([25, 25])
    hum_ors = [0, np.pi/4, np.pi/2, np.pi * 3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4]
    hum_or = random.choice(hum_ors)
    x_ = []
    y_ = []
    v_ = []
    oris = []
    x_.append(hum_pos[0])
    y_.append(hum_pos[1])
    oris.append(hum_or)
    v_.append(1)
    for i in range(500):
        ag_pos = np.random.uniform(0, env.max_x, 2)
        x_.append(ag_pos[0])
        y_.append(ag_pos[1])
        ag_or = random.choice(hum_ors)
        state = np.concatenate([
            ag_pos - hum_pos , [hum_or], [ag_or]
        ])
        state = torch.FloatTensor(state).unsqueeze(0).to(device=device)
        value = evaluate_state(model, state, policy=policy)
        v_.append(value.detach().cpu().numpy())
        oris.append(ag_or)
    
    first = True
    for xi, yi, value, ori in zip(x_, y_, v_, oris):
        if first:
            marker_style = mmarkers.MarkerStyle(marker=6, fillstyle='full')
            transform = Affine2D().rotate_deg((ori * 180 / np.pi) + 90)
            marker_style._transform = transform

            ax.scatter(xi, yi, s=50, marker=marker_style, color='red')
            first = False
        else:
            marker_style = mmarkers.MarkerStyle(marker='^', fillstyle='full')
            transform = Affine2D().rotate_deg(ori * 180 / np.pi + 90)
            marker_style._transform = transform

            ax.scatter(xi, yi, c=[value], cmap='viridis', s=5, marker=marker_style, vmin=min(v_), vmax=max(v_))

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(v_), vmax=max(v_)))
    plt.colorbar(sm, label='Value', ax=ax)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Human Orientation: {hum_or*180/np.pi}')
    plt.savefig(f'./{name}.png')
    plt.close()


# def main():
#     env = Environment()
#     model = load_model(env=env, model_path='/home/sahar/Follow-ahead-3/DQN_test/a2c_navigation_or_random', policy='a2c')
#     plot_evaluate_states(env, model, policy='a2c')
#     evaluate_agent(env=env, model=model, policy='a2c')

# main()