# Group Relative Policy Optimization (GRPO) for Continuous Control

This document provides a mathematical description of the Off-Policy GRPO agent implemented in this repository.

## 1. Mathematical Formulation

The agent optimizes a stochastic policy $\pi_\theta(a|s)$ using an off-policy framework that combines group-sampled Q-estimates with KL-regularization and entropy maximization.

### A. The Actor Objective
The policy $\pi_\theta$ is updated to maximize the following objective function using the **Reparameterization Trick**:

$$J(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \xi \sim \mathcal{N}} \left[ \frac{1}{G} \sum_{i=1}^G Q(s, a_i(\theta, \xi)) \right] - \beta D_{KL}(\pi_\theta \| \pi_{ref}) + \eta H(\pi_\theta)$$

Where:
- $G$ is the **group size** (number of actions sampled per state).
- $Q(s, a_i)$ is the minimum of the double Q-critics.
- $\pi_{ref}$ is the **Reference Policy** (a slowly updated copy of the actor).
- $\beta$ is the KL penalty coefficient.
- $\eta$ is the entropy bonus coefficient.

### B. Group-Based Optimization
While standard GRPO for LLMs uses group-relative advantages to remove the critic, this continuous control implementation uses a **Group-Sampled Actor-Critic** approach. For each state $s$ in a batch, we sample $G$ actions. This provides:
1.  **Lower Variance**: Multiple samples per state provide a more stable gradient of the value surface.
2.  **Exploration**: The ensemble of actions encourages the policy to explore the Q-surface broadly around the current mean.

The code computes group-relative advantages for monitoring:
$$A(s, a_i) = \frac{Q(s, a_i) - \mu(Q)}{\sigma(Q) + \epsilon}$$
However, for the reparameterized gradient update, the raw mean of $Q$ is maximized, as the $\mu(Q)$ baseline cancels out in the gradient and $\sigma(Q)$ is handled by the optimizer's adaptive learning rate (e.g., Adam).

### C. Continuous Control via Reparameterization
To optimize through the continuous action space, we use a **Squashed Gaussian Policy**. Action $a$ is sampled as:
$$a = \text{tanh}(x), \quad x \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$$
This allows gradients to flow directly from the Q-function into the policy parameters $\theta$:
$$\nabla_\theta J(\theta) \approx \nabla_\theta \min(Q_1(s, \text{tanh}(x_\theta)), Q_2(s, \text{tanh}(x_\theta)))$$

## 2. Critic Architecture (Double Q-Learning)

The agent maintains two critics $Q_{\phi_1}, Q_{\phi_2}$ updated using **$n$-step Temporal Difference (TD)** targets:

$$y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \min_{j=1,2} Q_{target, j}(s_{t+n}, a')$$
Where $a'$ is sampled from the current policy $\pi_\theta$ with target policy smoothing.

## 3. Regularization Mechanisms

### KL Divergence Penalty
The KL penalty $D_{KL}(\pi_\theta \| \pi_{ref})$ prevents the policy from deviating too far from the reference policy. The reference policy $\pi_{ref}$ is updated using a periodic soft update:
$$\theta_{ref} \leftarrow (1-\tau_{ref})\theta_{ref} + \tau_{ref}\theta$$
(Typically $\tau_{ref} = 0.05$ every 100 updates).

### Target Policy Smoothing
During critic updates, we add clipped noise to the target action:
$$a' = \text{tanh}(\mu_\theta(s_{t+n}) + \epsilon), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \tilde{\sigma}), -c, c)$$
This prevents the policy from exploiting sharp, non-smooth peaks in the Q-function.
