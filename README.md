# ReinforcedTutor for Recommending Question Difficulty

## 🚀 Benefits for Users
This API helps **students optimize their study time** by:
- **Personalized recommendations** using reinforcement learning
- **Efficient question selection** based on previous performance data
- **Adaptive difficulty progression** preventing wasted effort
- **Confidence building** through strategic challenge scaling
- **Smart preparation** for exams with mixed-difficulty questions

Designed as the **AI core** for an educational app
- handles the recommendation logic while the app handles UI/question delivery.

## Workflow Diagram
              --------------------------
     +---------------+                  +---------------------+
     | 1. Take Initial|                  |                     |
     |    Assessment  |<-----------------| Question Bank       |
     +---------------+                  | (By Difficulty)     |
             |                          +---------------------+
             |                                  |
             v                                  v
     +---------------+                  +---------------------+
     | 2. Enter Test  |                  | Reinforcement       |
     | Scores & Needs |----------------->| Learning API        |
     +---------------+   User History    | (This Project)      |
             |                          +---------------------+
             |                                  |
             v                                  |
     +---------------+                  +---------------------+
     | 3. Get Question|                  | Study Interface    |
     | Recommendations|----------------->| (Displays Questions|
     +---------------+   Difficulty List +---------------------+
             |
             v
     +---------------+                    +-------------------------------------+
     | 4. Practice   |                    |                                     |
     | with Smart    |   ---------------> | Results are saved for future inputs |
     | Recommendations                    +-------------------------------------+
     +---------------+

## 🔍 How It Works

### Input Requirements:
```python
{
  "difficulty_scores": {
    "0": {"obtained": 95, "total": 100},  # Easiest
    "1": {"obtained": 85, "total": 100},
    "2": {"obtained": 70, "total": 100},
    "3": {"obtained": 60, "total": 100},
    "4": {"obtained": 40, "total": 100}   # Hardest
  },
  "questions_needed": 20
}
```
### Sample Output

``` --- Recommended Difficulty Sequence ---
For 20 questions: [4, 3, 1, 1, 4, 4, 3, 4, 0, 3, 3, 0, 4, 3, 1, 4, 1, 1, 4, 4]
```

## 🎓 Training the Model

After cloning the repo please install the required packages with
```bash
pip install -r requirements.txt
```
### Quick Start Training
```bash
# Train model with default settings (saves to models/agent.pth)
python src.train.py

# Train with progress visualization
python src.train --plot
```
Note: The model was trained on synthetic data. For better performance, it would be
better to train the model on real data.

## Evaluation
```bash
python src.main
```

### Student Model (Knowledge Tracking)
This model shows the student's knowledge progression. For difficulty level i \
Probability update when correct is:

Correct Answer Update:

$p_i^{(t+1)} = p_i^{(t)} + c \cdot \alpha p_i(1-p_i)$

Incorrect Answer Update:

$p_i^{(t+1)} = p_i^{(t)} - w \cdot \alpha p_i(1-p_i)$

Original Paper Equations (Malpani et al.)

Where:
* alpha = learning rate
* c = correct coefficient
* w = wrong coefficient
* N = total difficulty types

* In this class, we maintain probabilities for different question types.
* We update probabilities using a logistic-like curve (X term is a derivative of sigmoid)
* Harder questions get bigger boosts for correct answers but smaller penalties for wrong ones.

### Actor-Critic Network Architecture
This is the neural network that makes decisions (actor) and evaluates the states (critic).

**Architecture Diagram**

Input (State)\
│\
└─ Shared Layers (ReLU → Tanh)\
   ├─ Actor Head (Softmax) → Action Probabilities\
   └─ Critic Head (Tanh) → State Value

The shared layers learn general features, and the actor outputs the probability distribution over actions using softmax:
$\pi(a|s) = \frac{e^{z_a}}{\sum_{b}e^{z_b}}$
The critic estimates the state value using a tanh activation:

$V(s) = \tanh(w^T h + b)$

where h is the hidden layer output.

The Actor-Critic algorithm combines policy optimization with value estimation.

Advantage:\
The advantage measures how much better an action was compared to the critic's expectation.
* Positive advantage = action was better than average
* Negative advantage = action was worse than average

$$
A(s_t,a_t) = \sum_{k=0}^{T-t} \gamma^k\, r_{t+k} - V(s_t)
$$

In the code, advantage was returns - values where returns was the discounted cumulative rewards.

Policy Loss:\
This adjusts the Actor network to favor actions that lead to higher than expected rewards.
$L_{\pi} = -\frac{1}{T} \sum_{t=0}^T \log\pi(a_t|s_t) \cdot A(s_t,a_t)$

Value Loss:\
Trains the critic to better estimate state values
$L_{V} = \frac{1}{T} \sum_{t=0}^T (V(s_t) - \text{Actual Return})^2$

### Environment (TutoringEnv)

Simulates the tutoring process and calculates rewards

$$
\text{State} = \Big[p_1,\; p_2,\; p_3,\; p_4,\; p_5,\; \frac{\text{current step}}{\text{total steps}},\; 1 - \frac{\text{current step}}{\text{total steps}}\Big]
$$

where $p_i$ = probability for difficulty i

### Reward Function

$$
R = 5(a+1) + 10(p_{\text{new}} - p_{\text{old}}) + 6c +
\begin{cases}
0.5(1 - \sigma), & \text{if } n \ge 5 \\
0, & \text{if } n < 5
\end{cases}
$$

where:


- $ a $: Action (difficulty level selected by the agent).

- $ p_{\text{new}} $: Updated success probability after the action.

- $ p_{\text{old}} $: Previous success probability before the action.

- $ c $: Binary correctness indicator ($ 1 $ for correct, $ 0 $ for incorrect).

- $ n $: Number of stored difficulty levels (from \texttt{self.difficulties}).

- $ \sigma $: Standard deviation of difficulties (\texttt{np.std(self.difficulties)}).

The function incentivizes harder questions via $ 5(a+1) $, rewards improvement
through $ 10(p_{\text{new}} - p_{\text{old}}) $, grants $ 6c $ for correct answers,
and encourages consistency by penalizing variability in difficulties when $ n \geq 5 $.


### Gradient Policy Theorem
The core update rule for policy parameters $\theta$:

$$
\nabla J(\theta) \approx \mathbb{E}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot A(s_t,a_t)\right]
$$

where $A(s_t,a_t)$ is the advantage function.

### Value Function Update

The critic's temporal difference update:
$V(s) \leftarrow V(s) + \alpha\left[\sum_{k=0}^\infty \gamma^k r_{t+k} - V(s)\right]$

### Entropy Regularization
Encourages exploration through:
$H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)$

## Hyperparameters

**Hyperparameters (changable) and Their Roles**

| Parameter         | Value | Purpose                                                                 |
|-------------------|-------|-------------------------------------------------------------------------|
| $\gamma$ (gamma)  | 0.99  | Discount factor for future rewards                                     |
| $\alpha$ (alpha)  | 0.1   | Student model's learning rate                                         |
| `entropy_coeff`   | 0.01  | Strength of entropy regularization                                   |
| $c$ (correct)     | 0.2   | Probability boost magnitude for correct answers                     |
| $w$ (wrong)       | 0.05  | Probability reduction for incorrect answers                         |

## Action Selection

**Action Selection Process**
```python
def get_action(self, state):
    probs, value = self.model(state)  # Get network outputs
    probs = 0.9*probs + 0.1*uniform  # Add exploration noise
    dist = Categorical(probs)        # Create distribution
    action = dist.sample()           # Sample action
    return action, dist.log_prob(action), value, dist.entropy()
```

## Evaluation Metrics

**Key performance indicators:**
- Final student proficiencies per question type:
  $\mathbf{p} = [p_1, p_2, p_3, p_4, p_5]$
- Average total reward across trials:
  $\bar{R} = \frac{1}{N}\sum_{i=1}^N \left( \sum_{t=0}^T r_t^{(i)} \right)$
- Action distribution statistics:
  $\text{std}(\pi), \text{entropy}(\pi)$

## Implementation Balance

The system combines:
- **Student modeling**: Spaced repetition principles
- **Deep RL**: Actor-Critic with stabilization techniques
- **Exploration strategies**: Entropy regularization + noise injection
- **Curriculum learning**: Difficulty-based rewards

The Average total reward is a metric that gives insight into Agent Effectiveness, Training Progress and System Stability (depending on the standard deviation across trials)

### Training Process

1. Agent receives state containing student's current knowledge
2. Actor network selects question type (action)
3. Environment provides question and evaluates answer
4. Student model updates knowledge probabilities
5. New state and reward are calculated
6. Episode repeats for 20 questions
7. Agent updates network using all episode experiences
