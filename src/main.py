from collections import deque
from typing import List, Dict, Tuple
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import torch

from .environment import Student
from .model_builder import ActorCritic

class AdaptiveTester:
    def __init__(
        self,
        model_path: str,
        num_difficulty_levels: int = 5
    ):
        """
        Class that adapts test questions based on learned faetures.
        Args:
            model_path: Path to the trained Pytorch model
            num_difficulty_levels: Number of difficulty levels
        """
        self.levels = num_difficulty_levels
        self.episode_length_during_training = 100
        self.model = self._load_model(model_path)
        self.reset()

    def reset(self):
        self.student = Student(num_difficulty_levels=self.levels)
        self.correct_history = deque(maxlen=100)
        self.level_history = deque(maxlen=100)

    def add_initial_assessments(self, results: Dict[int, Tuple[int, int]]):
        for level, (corrects, total) in results.items():
            for _ in range(total):
                is_correct = corrects > 0
                self.student.update(difficulty=level, correct=is_correct)
                self.correct_history.append(is_correct)
                self.level_history.append(level)
                if corrects > 0: corrects -= 1

    def _calculate_performance(self):
        weights = np.array(self.level_history) + 1
        return np.dot(self.correct_history, weights)/ weights.sum()

    def _calculate_adjusted_progress(self, num_questions: int)-> float:
        questions_to_episode = num_questions/self.episode_length_during_training
        return min(questions_to_episode, 1)

    def _load_model(self, model_path: str):
        model = ActorCritic(
            input_size=self.levels + 4, # depends on features added
            hidden_size=256,
            output_size=self.levels
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    def _trace_level_change(self):
        if len(self.level_history) < 2:
            return 0.0
        x = np.arange(len(self.level_history))
        return Polynomial.fit(x, self.level_history, 1).convert().coef[1]

    def _create_state(self, progress) -> torch.Tensor:
        probs = self.student.probs
        return torch.tensor(
            np.concatenate([
                probs,
                [progress, 1 - progress],
                [self._calculate_performance(), self._trace_level_change()]
                ]),
            dtype=torch.float32
        )

    def suggest_next_question_levels(
        self,
        results: Dict[int, Tuple[int, int]],
        num_questions: int=10
    ) -> List[int]:
        self.reset()
        self.add_initial_assessments(results)

        question_difficulties = []
        with torch.no_grad():
            for s in range(num_questions):
                max_progress = self._calculate_adjusted_progress(num_questions)
                current_progress = s/num_questions
                adjusted_progress = current_progress * max_progress
                state = self._create_state(adjusted_progress)

                policy, _ = self.model(state.unsqueeze(0))
                action = int(torch.multinomial(policy, 1).item())
                question_difficulties.append(action)
        return question_difficulties

if __name__ == "__main__":
    results = {}
    num_difficulty_levels = 5

    print("\n--- Initial Assessment Input ---")
    print("For each difficulty level, enter:")
    print(" Marks Awarded [SPACE] Total Marks Available")
    print("Example: '85 100' (85 out of 100 correct)\n")

    for d in range(num_difficulty_levels):
        while True:
            try:
                entry = input(f"Difficulty {d} Results (correct/total): ")
                parts = entry.strip().split()
                if len(parts) != 2:
                    raise ValueError("Please enter exactly two numbers")
                corrects, total = map(int, parts)
                corrects = float(corrects)
                if total <= 0:
                    raise ValueError
                if corrects < 0 or corrects > total:
                    raise ValueError
                results[d] = (corrects, total)
                break # break the inner while loop
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again")

    while True:
        try:
            num_questions = int(input("\nNumber of questions to suggest: "))
            if num_questions <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive integer")
    model_path = "models/agent.pth"
    tester = AdaptiveTester(
        model_path=model_path,
        num_difficulty_levels=num_difficulty_levels
    )

    next_difficulties = tester.suggest_next_question_levels(
        results,
        num_questions=num_questions
    )
    print("\n--- Recommended Difficulty Sequence ---")
    print(f"For {num_questions} questions: {next_difficulties}")
