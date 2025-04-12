from typing import Tuple
import torch.nn as nn
from torch import Tensor

class ActorCritic(nn.Module):
    """
        Actor-Critic architecture for reinforcement learning.

        Implements shared feature extraction with separate policy (actor) and
        value (critic) heads. Uses:
        - Tanh activation for stable value estimation
        - Softmax policy for discrete action distributions
        - Shared backbone for feature reuse

        Args:
            input_size (int): Dimensionality of input state
            hidden_size (int): Number of units in hidden layers
            output_size (int): Number of possible actions (difficulty levels)

        Attributes:
            shared (nn.Sequential): Shared feature extraction network
            actor (nn.Sequential): Policy head producing action distributions
            critic (nn.Sequential): Value head estimating expected returns
    """
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ):
        super().__init__()

        # shared layers for actor and critic
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # actor head outputs a probability distribution over actions
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

        # critic head estimates the expected return of the current states
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )

    def forward(self, state: Tensor)-> Tuple[Tensor, Tensor]:
        """
            Implements the forward pass
            Args:
                state (torch.Tensor): Input state tensor (shape: [batch, input_size])

            Returns:
                Tuple containing:
                    - policy (torch.Tensor): Action probability distribution
                    - value (torch.Tensor): Estimated state value

            Network Flow:
            1. Shared feature extraction
            2. Policy head produces π(a|s) via softmax
            3. Value head produces V(s) estimate
        """
        x = self.shared(state)
        # \pi(a|s; \theta)
        policy = self.actor(x)
        # V(s; \psi)
        value = self.critic(x)

        return policy, value
