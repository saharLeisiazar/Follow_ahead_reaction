from stable_baselines3 import DQN
from nav_env import Environment
import torch
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