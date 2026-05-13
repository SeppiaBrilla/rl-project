# Group Relative Policy Optimization (GRPO) for Continuous Control

This document describes the implementation details, hyperparameters, and the core logic behind the Off-Policy GRPO agent.

## 1. The Core Idea

The objective of this implementation is to adapt the **Group Relative Policy Optimization (GRPO)** algorithm—originally designed for discrete, on-policy training in Large Language Models—for **continuous control tasks** in Reinforcement Learning.

Standard GRPO avoids a separate value network by comparing multiple completions for the same state. Our implementation evolves this by integrating **off-policy learning mechanics** (inspired by SAC and TD3) to handle the sample-efficiency requirements of robotic and physical simulations.

### Key Components:
- **Group-Relative Advantages**: For each state sampled from the buffer, the actor generates a group of $G$ actions. Instead of using the absolute Q-value, the advantage for each action is calculated as its deviation from the group mean:
  $$A_i = \frac{Q(s, a_i) - \text{mean}(Q(s, \text{group}))}{\text{std}(Q(s, \text{group}))}$$
- **KL Regularization**: To maintain stability, we regularize the current policy against a "reference policy" (a slowly updating version of the actor).
- **Off-Policy foundation**: Uses a Replay Buffer and Double Q-Critics to provide stable value estimates.

## 2. Hyperparameters Explained

### Core GRPO Parameters
*   **`group_size` (G)**: The number of actions sampled per state during the actor update. GRPO relies on this "group" to compute relative advantages. A larger group provides a more stable advantage estimate.
*   **`beta` ($\beta$)**: The weight of the **KL Divergence penalty**. A **higher beta means more penalty**, making the policy updates more conservative.
*   **`entropy_coef`**: Controls the **Entropy Bonus**, encouraging the policy to maintain exploration and preventing the standard deviation from collapsing too quickly.

### Off-Policy & Critic Parameters
*   **`n_step`**: Horizon for **Multi-step TD returns**. It looks $n$ steps ahead to calculate rewards, accelerating learning and reducing value estimation bias.
*   **`tau` ($\tau$)**: The **Soft Update** rate for target networks (target critics and reference policy).
*   **`learning_starts`**: The number of random steps collected before training begins.
*   **`batch_size`**: The number of transitions sampled from the `ReplayBuffer` per gradient step.

### Optimization & Stability
*   **`lr` (Learning Rate)**: The step size for the Adam optimizer (usually `3e-4`).
*   **`gamma` ($\gamma$)**: The **Discount Factor** for future rewards.
*   **`max_grad_norm`**: Threshold for **Gradient Clipping** to prevent training instability.
*   **`policy_noise` & `noise_clip`**: Used for **Target Policy Smoothing**, adding noise to target actions during critic updates to make the value function more robust.

## 3. Implementation Details

- **Squashed Gaussian Policy**: We use a `tanh` activation on the actor's output to keep actions bounded within `[-1, 1]`, with a Jacobian correction for log-probabilities.
- **Update Frequency**: The implementation maintains a **1:1 update-to-step ratio** (performing `n_envs` updates per environment step) to match the efficiency of SAC/TD3.
- **Vectorization**: Action sampling and critic evaluations are fully vectorized using PyTorch to ensure high FPS on both GPU and CPU.
