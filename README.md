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
Probability update when correct is:\
\begin{equation}
p_i^{(t+1)} = \min\left(p_i^{(t)} + c \cdot \underbrace{\alpha p_i(1-p_i)}_{X} \cdot \left(1+\frac{i}{N}\right), 0.95\right)
\end{equation}
Probability update when incorrect is:\
\begin{equation}
p_i^{(t+1)} = \max\left(p_i^{(t)} - w \cdot \underbrace{\alpha p_i(1-p_i)}_{X} \cdot \left(1-\frac{i}{2N}\right), 0.05\right)
\end{equation}
Original Paper Equations (Malpani et al.)

Correct Answer Update:
\begin{equation}
p_i^{(t+1)} = p_i^{(t)} + c \cdot \alpha p_i(1-p_i)
\end{equation}

Incorrect Answer Update:
\begin{equation}
p_i^{(t+1)} = p_i^{(t)} - w \cdot \alpha p_i(1-p_i)
\end{equation}

(In our approach, we added a difficulty modulation)
When the user fails a difficult question, the system penalizes the user less than failing an easy question as N increases. this is why the smoothening uses 2N as the denominator for punishment but only N when rewarding.

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

The shared layers learn general features, and the actor outputs the probability distribution over actions using softmax:\
\begin{equation}
\pi(a|s) = \frac{e^{z_a}}{\sum_{b}e^{z_b}}
\end{equation}
The critic estimates the state value using a tanh activation:
\begin{equation}
V(s) = \tanh(w^T h + b)
\end{equation}
where h is the hidden layer output.\

The Actor-Critic algorithm combines policy optimization with value estimation.

Advantage:\
The advantage measures how much better an action was compared to the critic's expectation.
* Positive advantage = action was better than average
* Negative advantage = action was worse than average

\begin{equation}
A(s_t,a_t) = \underbrace{\sum_{k=0}^{T-t} \gamma^k r_{t+k}}_{\text{Actual Return}} - \underbrace{V(s_t)}_{\text{Critic's Prediction}}
\end{equation}

In the code, advantage was returns - values where returns was the discounted cumulative rewards.

Policy Loss:\
This adjusts the Actor network to favor actions that lead to higher than expected rewards.
\begin{equation}
L_{\pi} = -\frac{1}{T} \sum_{t=0}^T \log\pi(a_t|s_t) \cdot A(s_t,a_t)
\end{equation}

Value Loss:\
Trains the critic to better estimate state values
\begin{equation}
L_{V} = \frac{1}{T} \sum_{t=0}^T (V(s_t) - \text{Actual Return})^2
\end{equation}

### Environment (TutoringEnv)

Simulates the tutoring process and calculates rewards
\begin{equation}
\text{State} = [p_1, p_2, p_3, p_4, p_5, \underbrace{\frac{\text{current\_step}}{\text{total\_steps}}}_{\text{progress}}, \underbrace{1 - \frac{\text{current\_step}}{\text{total\_steps}}}_{\text{remaining}}]
\end{equation}
where $p_i$ = probability for difficulty i

### Reward Function
\begin{equation}
\text{Reward} = \underbrace{(a+1)\cdot0.2}_{\text{Difficulty bonus}} + \underbrace{10(p_{new}-p_{old})}_{\text{Improvement}} + \underbrace{2\cdot\text{correct}}_{\text{Correctness}}
\end{equation}

### Gradient Policy Theorem
The core update rule for policy parameters $\theta$:
\begin{equation}
\nabla J(\theta) \approx \mathbb{E}\left[\sum_{t=0}^T \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot A(s_t,a_t)\right]
\end{equation}
where $A(s_t,a_t)$ is the advantage function.

### Value Function Update

The critic's temporal difference update:
\begin{equation}
V(s) \leftarrow V(s) + \alpha\left[\sum_{k=0}^\infty \gamma^k r_{t+k} - V(s)\right]
\end{equation}

### Entropy Regularization
Encourages exploration through:
\begin{equation}
H(\pi) = -\sum_a \pi(a|s)\log\pi(a|s)
\end{equation}

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
