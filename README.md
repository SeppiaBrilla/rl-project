# RL-Project: Modular Reinforcement Learning Framework

A flexible and extensible Reinforcement Learning framework designed for training agents on diverse environments, including standard Gymnasium, Gymnasium Box2D, and DeepMind Control Suite (DMC).

## 🚀 Key Features

- **Multi-Suite Support**: Seamlessly switch between Gymnasium and DeepMind Control environments.
- **DMC Integration**: Automatic conversion of DMC's complex `Dict` observations into flat vectors for easier training.
- **Advanced Algorithms**: Built-in support for state-of-the-art RL algorithms including PPO, SAC, and TD3.
- **CNN Integration**: Optimized for vision-based tasks (e.g., `CarRacing-v3`) using NatureCNN and specialized observation wrappers.
- **Efficient Preprocessing**: Automatic grayscaling, resizing (84x84), and frame stacking (4 frames) for image environments to boost training speed by up to 16x.
- **Model Serialization**: easily save and load trained agents with a single flag.
- **Customizable Agents**: Abstract base classes for easily implementing and swapping new RL algorithms.
- **Flexible Buffers**: Adaptive `ReplayBuffer` for off-policy algorithms and `RolloutBuffer` for on-policy algorithms (e.g., PPO).

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

**Standard Gymnasium (CartPole):**
```bash
python train.py --env CartPole-v1 --episodes 500
```

**Gymnasium Box2D (CarRacing) with Model Saving:**
```bash
python train.py --env CarRacing-v3 --episodes 100 --save-model car_racing_ppo.pt
```

**DeepMind Control Suite (Cartpole Swingup):**
```bash
python train.py --env dm_control/cartpole-swingup-v0 --episodes 200
```

### CLI Arguments
- `--env`: Environment ID (standard Gym ID or `dm_control/[domain]-[task]-v0`).
- `--algo`: Algorithm to use. Choices: `DQN`, `SAC`, `TD3`, `PPO` (default: `DQN`).
- `--episodes`: Number of training episodes.
- `--render`: Enable visual rendering (human mode).
- `--seed`: Set random seed for reproducibility.
- `--save-model`: Path to save the model weights and optimizer state upon completion (e.g., `model.pt`).

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
- Use `src/env/factory.py` for all environment creation to ensure consistent wrapping.
- All agents **must** implement the `BaseAgent` interface.
- Standard off-policy agents (DQN, SAC, TD3) use `ReplayBuffer`, while on-policy algorithms (PPO) use `RolloutBuffer` providing GAE advantages and returns. Ensure you maintain this distinction in `train.py`.
