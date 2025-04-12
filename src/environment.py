from collections import deque
from typing import Optional, Tuple, Dict, Any, cast
import random
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from datasets import Dataset

class Student:
    """
        Model representing student knowledge across difficulty levels.

        Implements a probabilistic skill update mechanism with:
        - Difficulty-dependent learning rates
        - Performance-based probability adjustments
        - Adaptive difficulty response

        Args:
            num_difficulty_levels (int): Number of difficulty levels
            alpha (float): Base learning rate (0.1)
            c (float): Correct answer update coefficient (0.6)
            w (float): Wrong answer update coefficient (0.15)
            initial_probs (Optional[np.ndarray]): Initial probabilities
                of success per difficulty type

        Attributes:
            levels (int): Number of difficulty levels
            probs (np.ndarray): Success probability for each difficulty level
            alpha (float): Base learning rate
            c (float): Correct answer multiplier
            w (float): Wrong answer multiplier
        """
    def __init__(
        self,
        num_difficulty_levels: int,
        alpha: float=0.1,
        c: float=0.6,
        w: float=0.15,
        initial_probs: Optional[np.ndarray] = None
    ):
        self.levels = num_difficulty_levels
        self.alpha = alpha
        self.c = c
        self.w = w

        if initial_probs is None:
            base = np.random.uniform(0.5, 0.9)  # Varied starting point
            decline = np.random.uniform(0.1, 0.3)  # Random difficulty curve
            self.probs = np.linspace(
                start=base,
                stop=base-decline,
                num=num_difficulty_levels
            )
        else:
            assert len(initial_probs) == num_difficulty_levels
            self.probs = np.array(initial_probs)

    def __repr__(self):
        """Debug-friendly representation"""
        return f"StudentModel(levels={self.levels}, " \
               f"probs={[f'{p:.2f}' for p in self.probs]})"

    def __getitem__(self, difficulty):
        """Allow direct access to probabilities"""
        return self.probs[difficulty]

    def __len__(self):
        """Number of difficulty levels"""
        return self.levels

    def __eq__(self, other):
        """Compare student states"""
        return np.array_equal(self.probs, other.probs)

    def update(self, difficulty: int, correct: bool) -> None:
        """
        Update the student's knowledge state for difficulty level 'difficulty'
        according to the original formulation by Malpani et al.

        The update equations are:

        Correct answer:
            p_i^{(t+1)} = p_i^{(t)} + c * α * p_i * (1 - p_i)

        Incorrect answer:
            p_i^{(t+1)} = p_i^{(t)} - w * α * p_i * (1 - p_i)

        Args:
            difficulty (int): The difficulty level index to update.
            correct (bool): Indicates whether the response was correct.
        """
        # Retrieve the current probability for the given difficulty level.
        current_prob = self.probs[difficulty]

        # Calculate the base update term: X = α * p_i * (1 - p_i)
        X = self.alpha * current_prob * (1 - current_prob)

        # Update the probability based on whether the answer is correct.
        if correct:
            new_prob = current_prob + self.c * X  # Correct response update.
        else:
            new_prob = current_prob - self.w * X  # Incorrect response update.

        # Save the updated probability back to the model.
        self.probs[difficulty] = new_prob


class ExamEnv:
    """
        Reinforcement Learning environment for adaptive scenarios.

        Manages:
        - Question bank organization
        - Student state tracking
        - Reward calculation
        - State representation
        - Episode management

        Args:
            dataset (List[Dict]): Collection of questions with 'difficulty' and 'question' keys
            num_difficulty_levels (int): Number of difficulty categories (default: 5)
            questions_per_episode (int): Episode length in questions (default: 100)

        Attributes:
            student (Student): Current student model instance
            question_bank (List[deque]): Shuffled question pools per difficulty
            corrects (deque): History of recent correctness (20 steps)
            difficulties (deque): History of recent difficulties (20 steps)
    """
    def __init__(
        self,
        dataset: Dataset,
        num_difficulty_levels: int=5,
        questions_per_episode: int=100
    ):
        self.num_questions = questions_per_episode

        # history tracking
        self.corrects = deque(maxlen=20)
        self.difficulties = deque(maxlen=20)
        self.levels = num_difficulty_levels

        # Organize the questions by difficulty
        self.questions = [[] for _ in range(self.levels)]
        # debug: dataset must have 'difficulty' and 'question' as keys
        for item in dataset:
            data_item = cast(Dict[str, Any], item)
            difficulty = data_item['difficulty']
            assert isinstance(difficulty, int), (
                f"Invalid difficulty type for question: {data_item['question']}\n"
                f"Expected integer, got {type(difficulty).__name__} "
                f"(value: {difficulty})"
            )
            self.questions[difficulty].append(data_item['question'])
        self.student = None
        self.question_bank = None
        self.current_step = 0.0

    def __len__(self):
        """Number of questions per episode"""
        return self.num_questions

    def __getitem__(self, difficulty):
        """Direct question access for testing"""
        return self.questions[difficulty]

    def __contains__(self, difficulty):
        """Check if difficulty has questions"""
        return len(self.questions[difficulty]) > 0

    def __iter__(self):
        """Enable iteration over episode steps"""
        return self

    def __next__(self):
        if self.current_step >= self.num_questions:
            raise StopIteration
            return self.step(self.agent.get_action(self._get_state())[0])

    def reset(self)-> np.ndarray:
        """
        Resets the env for a new episode
        """
        self.current_step = 0.0
        self.student = Student(num_difficulty_levels=self.levels)
        # initialize a pool for questions in a deque object
        self.question_bank = [deque() for _ in range(self.levels)]
        self.corrects.clear()
        self.difficulties.clear()
        return self._get_state()

    def _get_state(self)-> np.ndarray:
        if not self.student:
            self.reset()
        assert self.student is not None
        # temporal awareness
        progress = float(self.current_step)/float(self.num_questions)
        performance = np.mean(self.corrects).item() if self.corrects else 0.5
        difficulty_trend = 0.0

        if len(self.difficulties) >= 2:
            x = np.arange(len(self.difficulties))
            p = Polynomial.fit(x, self.difficulties, 1)
            difficulty_trend = float(p.coef[1])
        return np.concatenate(
            [
                self.student.probs,
                [progress, 1 - progress],
                [performance, difficulty_trend]
            ]
        )

    def _get_question(self, difficulty: int)-> str:
        """
        Get next question of the selected difficulty
        """

        if not self.question_bank:
            self.reset()
        assert self.question_bank is not None

        if not self.question_bank[difficulty]:
            # refill question bank
            quests = self.questions[difficulty].copy()
            random.shuffle(quests)
            self.question_bank[difficulty].extend(quests)

        return self.question_bank[difficulty].popleft()

    def step(
        self, action: int, correct: Optional[bool]=None
    )-> Tuple[np.ndarray, float, bool, Dict[str, Any]]:

        if not self.student:
            self.reset()
        assert self.student is not None

        question = self._get_question(action)
        self.difficulties.append(action)

        old_prob_success = self.student.probs[action]
        if correct is None:
            # leave room for random chance if student is unevaluated
            correct = bool(np.random.rand() < old_prob_success)
        self.corrects.append(correct)
        # update student model
        self.student.update(action, correct)
        # rewards
        # incentive for harder questions
        difficulty_bonus = (action + 1) * 5
        correct_reward = 6 * correct
        # consistency reward
        consistency_reward = 0.0
        if len(self.difficulties) >= 5:
            std = np.std(self.difficulties)
            consistency_reward = float(0.5 * (1 - std))

        total_reward = (
            difficulty_bonus +
            correct_reward +
            consistency_reward
        )
        self.current_step += 1
        done = self.current_step >= self.num_questions

        return self._get_state(), total_reward, done, {'question': question}
