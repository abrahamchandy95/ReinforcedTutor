from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

@dataclass
class TrainConfig:
    """Main configuration for training"""
    num_episodes: int = 1000
    num_difficulty_levels: int = 5
    questions_per_episode: int = 100
    extra_features: int = 2
    plot: bool = False

@dataclass
class EpisodeRecord:
    """Single episode training metrics"""
    number: int
    total_reward: float
    action_counts: np.ndarray
    action_probs: Optional[np.ndarray] = None
    proficiency: Optional[np.ndarray] = None
    num_hardest_correct: Optional[float] = None

    def __post_init__(self):
        """Automatically calculate action probabilities"""
        if self.action_probs is None:
            total = self.action_counts.sum()
            if total == 0:
                self.action_probs = np.zeros_like(self.action_counts)
            else:
                self.action_probs = self.action_counts / total
        # Added runtime validation
        assert self.action_probs is not None

@dataclass
class TrainHistory:
    """Aggregate training metrics"""
    episodes: List[EpisodeRecord] = field(default_factory=list)

    def __post_init__(self):
        self.episodes = []

    def add_episode(
        self,
        episode_num: int,
        reward: float,
        action_counts: np.ndarray,
        proficiency: Optional[np.ndarray] = None,
        num_hardest_correct: Optional[float] = None
    ):
        """Add new training episode results"""
        episode = EpisodeRecord(
            number=episode_num,
            total_reward=reward,
            action_counts=action_counts,
            proficiency=proficiency,
            num_hardest_correct=num_hardest_correct
        )
        self.episodes.append(episode)

    def get_metrics(self):
        return {
            'rewards': [
                e.total_reward for e in self.episodes
            ],
            'action_distributions': [
                e.action_probs for e in self.episodes
            ],
            'proficiencies': [
                e.proficiency
                for e in self.episodes
                if e.proficiency is not None
            ],
            'num_hardest_correct': [
                e.num_hardest_correct
                for e in self.episodes
                if e.num_hardest_correct is not None
            ]
        }

@dataclass
class EvalResult:
    """Stores results from a single evaluation trial"""
    trial_number: int
    total_reward: float
    proficiency: np.ndarray
    num_hardest_correct: float
