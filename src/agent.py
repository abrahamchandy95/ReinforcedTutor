from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor

from src.model_builder import ActorCritic

class Tutor:
    def __init__(self, num_difficulty_levels: int, state_size: int):
        """
            Reinforcement Learning agent using Actor-Critic algorithm.
            Implements Proximal Policy Optimization (PPO) with:
            - Advantage estimation
            - Entropy regularization for exploration
            - Value function approximation
            - Gradient clipping for stable training

            Args:
                num_difficulty_levels (int): Number of difficulty levels
                state_size (int): Input Size of the Neural Network

            Attributes:
                gamma (float): Discount factor (0.97) for future rewards
                beta (float): Entropy regularization coefficient (0.05)
                model (ActorCritic): Neural network model
                optimizer (torch.optim.Adam): Optimizer
                levels (int): Number of difficulty levels
        """
        # discount factor close to 1 for long term goals
        self.gamma = 0.97
        # entropy regularization coefficient for exploration
        self.beta = 0.05

        self.model = ActorCritic(
            input_size=state_size,
            hidden_size=256,
            output_size=num_difficulty_levels
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.levels = num_difficulty_levels

    def __len__(self)-> int:
        """
            Returns:
                int: Number of question difficulty levels
        """
        return self.levels

    def get_action(
        self,
        state: np.ndarray
    )-> Tuple[int, Tensor, Tensor, Tensor]:
        """
            Sample action using current policy with exploration.

            Args:
                state (np.ndarray): Current state features
                    (shape: [state_size])

            Returns:
                tuple containing:
                    - int: Selected action (difficulty level)
                    - torch.Tensor: Log probability of selected action
                    - torch.Tensor: Value estimate for current state
                    - torch.Tensor: Policy entropy value
        """
        # convert state to tensor: s \in \mathbb{R}^d

        state_tensor: Tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        policy: torch.Tensor
        value: torch.Tensor
        policy, value = self.model(state_tensor)

        # add 10% exploration
        num_actions = policy.shape[-1]
        exploration = torch.full_like(policy, 1.0/num_actions)
        probs = 0.9 * policy + 0.1 * exploration
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return int(action.item()), dist.log_prob(action).squeeze(), value, dist.entropy()

    def update(
        self,
        log_probs: List[Tensor],
        values: List[Tensor],
        entropies: List[Tensor],
        rewards: List[float]
    )-> None:
        """
            Perform policy and value network update using PPO algorithm.

            Args:
                log_probs: List of action log probabilities from policy
                values: List of state value estimates
                entropies: List of policy entropy values
                rewards: List of observed rewards

            Training Process:
            1. Compute discounted returns using dynamic programming
            2. Calculateand normalizes advantages
            3. Combine losses and perform gradient update with clipping
        """
        # Calculate discounted returns using dynamic programming
        # R_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}
        returns = []
        discounted_reward = 0.0
        for r in reversed(rewards):
            discounted_reward = r + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns_tensor: Tensor = torch.tensor(returns, dtype=torch.float32)
        # concatenate value estimates V(s_0), V(s_1), ..., V(s_T)
        values_tensor: Tensor = torch.cat(values)

        # Advantage: A(s, a) = R - V(s)
        advantages: Tensor = returns_tensor - values_tensor
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # calculate policy loss
        # L_policy = -\mathbb{E} \left[ \log{\pi}(a|s)\hat{A}(s, a) ]\right
        l_policy: Tensor = (-torch.cat([lp.unsqueeze(0) for lp in log_probs]) * advantages.detach()).mean()
        # entropy loss for exploration
        # L_entropy = -\beta * \mathbb{E} [ H (\pi(. | s)) ]
        l_entropy: Tensor = -torch.cat(entropies).mean() * self.beta
        # value loss (MSE):
        l_value: Tensor = 0.5 * advantages.pow(2).mean()

        total_loss: Tensor = l_policy + l_entropy + l_value
        # gradient descent
        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
