import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(metrics, num_episodes):
    """Plot training metrics"""
    plt.figure(figsize=(15, 10))
    # Reward plot
    plt.subplot(2, 2, 1)
    rewards = metrics.get('rewards', [])
    if rewards:
        plt.plot(rewards)
        plt.title('Training Reward Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

    # Action distribution plot
    plt.subplot(2, 2, 2)
    action_dists = metrics.get('action_distributions', [])
    if action_dists:
        action_matrix = np.array(action_dists)
        for i in range(action_matrix.shape[1]):
            plt.plot(action_matrix[:, i], label=f'Q{i+1}')
        plt.title('Action Selection Distribution')
        plt.legend()

    # Proficiency plot
    plt.subplot(2, 2, 3)
    proficiencies = metrics.get('proficiencies', [])
    if proficiencies:
        proficiencies = np.array(proficiencies)
        episodes = np.linspace(0, num_episodes, len(proficiencies))
        for i in range(proficiencies.shape[1]):
            plt.plot(episodes, proficiencies[:, i], label=f'Q{i+1}')
        plt.title('Student Proficiency Development')
        plt.legend()

    # Hardest correct plot
    plt.subplot(2, 2, 4)
    hardest_correct = metrics.get('num_hardest_correct', [])
    if hardest_correct:
        plt.plot(hardest_correct)
        plt.title('Max Difficulty Correct Rate')

    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/training_metrics.png')
    plt.close()

def print_results(results):
    """Print formatted evaluation results"""
    avg_reward = np.mean([r.total_reward for r in results])
    std_reward = np.std([r.total_reward for r in results])
    avg_proficiency = np.mean([r.proficiency for r in results], axis=0)
    avg_max_diff = np.mean([r.num_hardest_correct for r in results])

    print("\nFinal Evaluation Results:")
    print(f"Average Total Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Max Difficulty Correct Rate: {avg_max_diff:.2%}")
    print("Average Proficiencies:")
    for i, p in enumerate(avg_proficiency):
        print(f"Q{i+1}: {p:.2%} ", end="")
