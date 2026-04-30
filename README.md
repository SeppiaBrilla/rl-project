# RL-Project: Modular Reinforcement Learning Framework

A flexible and extensible Reinforcement Learning framework designed for training agents on diverse environments, including standard Gymnasium, Gymnasium Box2D, and DeepMind Control Suite (DMC).

## 🚀 Key Features

- **Multi-Suite Support**: Seamlessly switch between Gymnasium and DeepMind Control environments.
- **DMC Integration**: Automatic conversion of DMC's complex `Dict` observations into flat vectors for easier training.
- **Advanced Algorithms**: Built-in support for state-of-the-art RL algorithms including PPO, SAC, and TD3.
- **CNN Integration**: Optimized for vision-based tasks (e.g., `CarRacing-v3`) using NatureCNN and specialized observation wrappers.
- **Vectorized Training**: Support for parallel environment execution using `SyncVectorEnv` and `AsyncVectorEnv` for significantly faster data collection.
- **Efficient Preprocessing**: Automatic grayscaling, resizing (84x84), and frame stacking (4 frames) for image environments to boost training speed by up to 16x, now fully compatible with batched vectorized observations.
- **Model Serialization**: Easily save and load trained agents with a single flag.
- **Customizable Agents**: Abstract base classes for easily implementing and swapping new RL algorithms, all supporting batched transitions.
- **Flexible Buffers**: Adaptive `ReplayBuffer` and `RolloutBuffer` (with GAE support) optimized for vectorized environment transitions.

---

## 📂 Project Structure

```bash
.
├── src/
│   ├── agents/           # RL Algorithm implementations
│   │   ├── base.py       # Abstract Base Agent (inherit from this)
│   │   ├── networks.py   # Shared neural network architectures (MLP, NatureCNN)
│   │   ├── dqn.py        # DQN Implementation
│   │   ├── ppo.py        # PPO Implementation (On-Policy)
│   │   ├── sac.py        # SAC Implementation (Off-Policy, Entropy Tuning)
│   │   └── td3.py        # TD3 Implementation (Off-Policy, Twin Critic)
│   ├── env/              # Environment management
│   │   ├── factory.py    # create_env(): The main entry point for envs
│   │   └── wrappers.py   # Custom Gymnasium wrappers
│   └── utils/            # Shared utilities
│       ├── buffer.py     # Adaptive Replay Buffer
│       ├── logger.py     # Standardized logging
│       └── ...
├── train.py              # Main execution script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.10+
- `swig` (needed for Box2D environments like CarRacing)
- `mujoco` (handled via `dm_control`)

### 2. Setup Environment
It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📖 Usage

Train an agent using the `train.py` script.

### Examples

**Standard Gymnasium (CartPole) with 4 parallel environments:**
```bash
python train.py --env CartPole-v1 --epochs 500 --n-envs 4
```

**Gymnasium Box2D (CarRacing) with PPO and Model Saving:**
```bash
python train.py --env CarRacing-v3 --algo PPO --epochs 100 --save-model car_racing_ppo.pt --n-envs 8
```

**DeepMind Control Suite (Acrobot Swingup) with SAC:**
```bash
python train.py --env dm_control/acrobot-swingup-v0 --algo SAC --epochs 200 --results-file acrobot_sac.csv
```

### Evaluation & Visualization

Once training is complete, you can visualize the trained agent's performance using `evaluate.py`. This script loads the saved model weights and runs the agent in "human" rendering mode.

**Example for CarRacing:**
```bash
python evaluate.py --env CarRacing-v3 --algo PPO --model-path car_racing_ppo.pt --episodes 5
```

**Example for CartPole:**
```bash
python evaluate.py --env CartPole-v1 --algo PPO --model-path model.pt --episodes 10
```

### `train.py` Arguments
- `--env`: Environment ID (standard Gym ID or `dm_control/[domain]-[task]-v0`).
- `--algo`: Algorithm to use. Choices: `DQN`, `SAC`, `TD3`, `PPO` (default: `DQN`).
- `--epochs`: Number of training epochs (rollouts or update cycles).
- `--n-envs`: Number of parallel environments to run (default: 1).
- `--render`: Enable visual rendering (human mode). Note: rendering with many parallel envs can be slow.
- `--seed`: Set random seed for reproducibility.
- `--save-model`: Path to save the model weights (`.pt`).
- `--results-file`: Path to save training progress CSV (default: `results.csv`).

### `evaluate.py` Arguments
- `--env`: Environment ID.
- `--algo`: Algorithm type (`SAC`, `TD3`, `PPO`, `DQN`).
- `--model-path`: Path to the `.pt` file containing trained weights.
- `--episodes`: Number of evaluation episodes (default: 5).
- `--seed`: Random seed.
- `--no-render`: Disable visualization (run headless).

---

## 🔧 Extending the Framework

### Adding a New Algorithm
1. Create a new file in `src/agents/` (e.g., `ppo.py`).
2. Inherit from `BaseAgent` in `src/agents/base.py`.
3. Implement `select_action` and `update` logic.
4. Register the new algorithm in `train.py`.

### Adding a New Environment Suite
The environment factory in `src/env/factory.py` is designed to be extensible:
```python
# src/env/factory.py

def create_env(env_id: str, ...):
    if env_id.startswith("my_new_suite/"):
        # Handle custom registration or loading logic here
        pass
    ...
```

---

## 📝 Reference Documents
- `DM887_Project.pdf`: Project description and requirements.

---

## 🤖 LLM Implementation Notes
When maintaining or extending this codebase:
- Use `src/env/factory.py` (`create_vector_env`) for environment creation to ensure consistent wrapping and vectorization.
- All agents **must** implement the `BaseAgent` interface and handle **batched** observations and actions to support vectorized training.
- Standard off-policy agents (DQN, SAC, TD3) use `ReplayBuffer`, while on-policy algorithms (PPO) use `RolloutBuffer` providing GAE advantages and returns. 
- The training loop in `train.py` is epoch-based, where each epoch involves collecting data from `n-envs` and performing agent updates.
