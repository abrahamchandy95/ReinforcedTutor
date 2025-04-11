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
