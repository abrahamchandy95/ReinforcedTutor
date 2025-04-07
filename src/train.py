"""
Adaptive Tutoring System Training Pipeline

This script:
1. Handles command-line arguments
2. Loads and preprocesses datasets
3. Trains the RL tutoring agent
4. Evaluates model performance
5. Saves results and visualizations

Usage:
    python train.py [--plot]

Optional Arguments:
    --plot   Generate training visualization plots
"""

import argparse
import random
from typing import cast
import os
import torch
from datasets import load_dataset, DatasetDict, Dataset

from src.config import TrainConfig
from src.engine import Engine
from src.utils import plot_metrics, print_results

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with:
            plot (bool): Whether to generate training plots
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot', action='store_true', help='Enable training plots'
    )
    return parser.parse_args()

def add_difficulty(data):
    """
    Dataset preprocessing function to add synthetic difficulty labels.
    Args:
        data (dict): Individual data sample
    Returns:
        dict: Original data with added 'difficulty' key (0-4)

    Note: This is a placeholder implementation. For real use cases,
    implement proper difficulty assessment based on problem complexity.
    """
    data['difficulty'] = random.randint(0, 4)
    return data

if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        plot=args.plot
    )

    dataset: DatasetDict = cast(
        DatasetDict, load_dataset("gsm8k", "main", streaming=False)
    )
    assert isinstance(dataset, DatasetDict)
    train_data: Dataset = dataset["train"].map(add_difficulty)
    test_data: Dataset = dataset["test"].map(add_difficulty)
    engine = Engine(config)

    agent, history = engine.train(dataset=train_data)
    # Save the model
    os.makedirs('results', exist_ok=True)
    torch.save(agent.model.state_dict(), "models/agent.pth")
    eval_results = engine.evaluate(dataset=test_data, num_trials=100)

    # Print evaluation summary
    print_results(eval_results)

    if config.plot:
        plot_metrics(history.get_metrics(), config.num_episodes)
