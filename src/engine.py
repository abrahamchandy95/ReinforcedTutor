from typing import Tuple, List, Optional
import torch
import numpy as np
from datasets import Dataset

from src.environment import ExamEnv
from src.agent import Tutor
from src.config import TrainConfig, TrainHistory, EvalResult

class Engine:
    """
        Core training and evaluation logic for adaptive tutoring.

        Manages interaction between environment (ExamEnv)
        and agent (Tutor).

        Args:
            config (TrainConfig): Object containing hyper-parameters

        Attributes:
            config (TrainConfig): Training configuration parameters
            _trained (bool): Flag indicating if training has been completed
            env (Optional[ExamEnv]): Training environment instance
            agent (Optional[Tutor]): Reinforcement learning agent instance
    """
    def __init__(self, config: TrainConfig):
        self.config = config
        self._trained = False
        self.env: Optional[ExamEnv] = None
        self.agent: Optional[Tutor] = None

    def train(self, dataset: Dataset)-> Tuple[Tutor, TrainHistory]:
        """
            Training process with policy optimization.

            Args:
                dataset: Training data containing question bank

            Returns:
                Tuple containing:
                    - Trained Tutor agent
                    - TrainHistory object with training metrics

            Training Process:
            1. Initialize environment and agent
            2. For each episode:
                a. Reset environment
                b. Collect experience through agent-environment interaction
                c. Update policy using collected experience
                d. Log metrics and progress
            3. Return trained agent and training history
        """
        input_size = (
            self.config.num_difficulty_levels +
            self.config.extra_features + 2
        )
        self.env = ExamEnv(
            dataset=dataset,
            num_difficulty_levels=self.config.num_difficulty_levels,
            questions_per_episode=self.config.questions_per_episode
        )
        self.agent = Tutor(
            num_difficulty_levels=self.config.num_difficulty_levels,
            state_size=input_size
        )
        history = TrainHistory()

        for episode in range(self.config.num_episodes):
            state = self.env.reset()
            episode_log_probs = []
            episode_values = []
            episode_rewards = []
            episode_entropies = []
            action_counts = np.zeros(
                self.config.num_difficulty_levels, dtype=np.int32
            )
            assert self.env.student is not None

            while True:
                # action selection with exploration
                action, log_prob, value, entropy = self.agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # store actions and other metrices
                action = int(action)
                action_counts[action] += 1
                episode_log_probs.append(log_prob)
                episode_values.append(value)
                episode_rewards.append(reward)
                episode_entropies.append(entropy)

                state = next_state
                if done:
                    hardest_corrects = self.env.student.probs[-1]
                    history.add_episode(
                        episode_num=episode,
                        reward=sum(episode_rewards),
                        action_counts=action_counts,
                        proficiency=self.env.student.probs.copy(),
                        num_hardest_correct=hardest_corrects
                    )
                    break
            # update policy
            self.agent.update(
                log_probs=episode_log_probs,
                values=episode_values,
                entropies=episode_entropies,
                rewards=episode_rewards
            )
            # print out whats happening
            if episode % 100 == 0:
                last_episode = history.episodes[-1]
                assert last_episode.action_probs is not None
                print(f"\nEpisode {episode} Action Distribution:")
                for i, p in enumerate(last_episode.action_probs):
                    print(f"Q{i+1}: {p:.2%} ", end="")

        self._trained = True
        assert self.agent is not None
        return self.agent, history

    def evaluate(
        self, dataset: Dataset, num_trials: int = 100
    )-> List[EvalResult]:
        """
            Evaluates the agent on a test dataset.

            Args:
                dataset: Evaluation dataset
                num_trials (int): Number of evaluation episodes

            Returns:
                List of objects containing:
                    - Total reward
                    - Proficiency metrics
                    - Maximum difficulty correctness

            Evaluation Workflow:
            1. Verify training completion
            2. Create evaluation environment
            3. For each trial:
                a. Reset environment
                b. Execute deterministic policy
                c. Collect evaluation metrics
        """
        if not self._trained:
            raise RuntimeError("Must call train() before evaluation")

        assert self.agent is not None
        env = ExamEnv(
            dataset=dataset,
            num_difficulty_levels=self.config.num_difficulty_levels,
            questions_per_episode=self.config.questions_per_episode
        )
        results = []
        for trial in range(num_trials):
            state = env.reset()
            episode_rewards = []
            while True:
                # select action (deterministic)
                with torch.no_grad():
                    action, _ , _, _ = self.agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                episode_rewards.append(reward)
                state = next_state
                if done:
                    assert env.student is not None
                    proficiency = env.student.probs.copy()
                    if len(proficiency) > 0:
                        hardest_corrects = proficiency[-1]
                    else:
                        hardest_corrects = 0.0
                    results.append(
                        EvalResult(
                            trial_number=trial,
                            total_reward=sum(episode_rewards),
                            proficiency=proficiency,
                            num_hardest_correct=hardest_corrects
                        )
                    )
                    break

        return results
