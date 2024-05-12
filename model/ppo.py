import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math
# from torch.distributions import Categorical
import numpy as np

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim=1526, policy_conv=True, action_std=0.1):
        super(ActorCritic, self).__init__()

        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=2, padding=(4, 4), bias=True),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                # # add for imagenet/AFHQ
                # nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=2, padding=(2, 2), bias=True),
                # nn.BatchNorm2d(24),
                # nn.ReLU()
            )

        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )

        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)

        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, 2),
            nn.Softmax(dim=1)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim / feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, restart_batch=False, training=False):
        if restart_batch:
            del memory.hidden[:]
            memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())

        if not self.policy_conv:
            state = state_ini.flatten(1)
        else:
            state = state_ini

        # # resize the image
        state = F.interpolate(state, size=(128, 128), mode='bicubic', align_corners=False)
        b, c, h, w = state.shape

        state = self.state_encoder(state)
        state = state.contiguous().view(b, -1)

        state, hidden_output = self.gru(state.view(1, state.size(0), state.size(1)), memory.hidden[-1])
        memory.hidden.append(hidden_output)

        state = state[0]
        action_mean = self.actor(state)

        dist = torch.distributions.Categorical(probs=action_mean)
        action = dist.sample().cuda()

        if training:
            action_logprob = dist.log_prob(action).cuda()
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_mean

        return action.detach()

    def evaluate(self, state, action):
        seq_l = state.size(0)
        batch_size = state.size(1)

        if not self.policy_conv:
            state = state.flatten(2)
            state = state.view(seq_l * batch_size, state.size(2))
        else:
            state = state.view(seq_l * batch_size, state.size(2), state.size(3), state.size(4))

        # resize the image  for CelebA dataset
        state = F.interpolate(state, size=(128, 128), mode='bicubic', align_corners=False)
        b, c, h, w = state.shape

        state = self.state_encoder(state)
        state = state.contiguous().view(b, -1)

        state = state.view(seq_l, batch_size, -1)

        state, hidden = self.gru(state, torch.zeros(1, batch_size, state.size(2)).cuda())
        state = state.view(seq_l * batch_size, -1)

        action_mean = self.actor(state)

        dist = torch.distributions.Categorical(probs=action_mean)

        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).cuda()
        dist_entropy = dist.entropy().cuda()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, policy_conv,
                 action_std=0.1, lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = 4

        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std).cuda()

        # FFHQ
        gen_path = './pretrain/FFHQ_pretrain/DDIM/ppo_model.pth'
        # imagenet
        # gen_path = './pretrain/ImageNet_premodels/ppo_model.pth'

        self.policy.load_state_dict(torch.load(gen_path), strict=True)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers, 1000, eta_min=0.0001, last_epoch=-1)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, restart_batch=False, training=True):
        return self.policy_old.act(state, memory, restart_batch, training)

    def update(self, memory):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.stack(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean()