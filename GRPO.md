# Group Relative Policy Optimization (GRPO) for Continuous Control

This document provides a mathematical description of the Off-Policy GRPO agent implemented in this repository.

## 1. Mathematical Formulation

The agent optimizes a stochastic policy $\pi_\theta(a|s)$ using an off-policy framework that combines group-relative advantages with KL-regularization.

### A. The Actor Objective
The policy $\pi_\theta$ is updated to maximize the following objective function:

$$J(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a_{i} \sim \pi_\theta} \left[ \frac{1}{G} \sum_{i=1}^G A(s, a_i) \right] - \beta \mathbb{E}_{s \sim \mathcal{D}} \left[ D_{KL}(\pi_\theta(\cdot|s) \| \pi_{ref}(\cdot|s)) \right] + \eta H(\pi_\theta)$$

Where:
- $G$ is the **group size** (number of actions sampled per state).
- $A(s, a_i)$ is the **Group-Relative Advantage**.
- $\pi_{ref}$ is the **Reference Policy** (a slowly updated copy of the actor).
- $\beta$ is the KL penalty coefficient.
- $\eta$ is the entropy bonus coefficient.

### B. Group-Relative Advantages
Unlike standard Reinforcement Learning which uses a value function $V(s)$ as a baseline, GRPO computes advantages by comparing actions within a sampled group. For a group of actions $\{a_1, a_2, \dots, a_G\}$ sampled for a single state $s$, the advantage for action $a_i$ is:

$$A(s, a_i) = \frac{Q(s, a_i) - \mu(Q)}{\sigma(Q) + \epsilon}$$

Where:
- $Q(s, a_i) = \min(Q_1(s, a_i), Q_2(s, a_i))$ (Double Q-Critic estimate).
- $\mu(Q) = \frac{1}{G} \sum_{j=1}^G Q(s, a_j)$ is the mean value of the group.
- $\sigma(Q)$ is the standard deviation of the group values.

### C. Continuous Control via Reparameterization
To efficiently optimize through continuous action spaces, we use the **Reparameterization Trick**. The action $a$ is sampled as:
$$a = \text{tanh}(\mu_\theta(s) + \sigma_\theta(s) \odot \xi), \quad \xi \sim \mathcal{N}(0, I)$$
This allows gradients to flow directly from the Q-function into the policy parameters $\theta$:
$$\nabla_\theta J(\theta) \approx \nabla_\theta \min(Q_1(s, \pi_\theta(s, \xi)), Q_2(s, \pi_\theta(s, \xi)))$$

## 2. Critic Architecture (Double Q-Learning)

The agent maintains two critics $Q_{\phi_1}, Q_{\phi_2}$ to combat overestimation bias. They are updated using **$n$-step Temporal Difference (TD)** targets:

$$y = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \min_{j=1,2} Q_{target, j}(s_{t+n}, a')$$
Where $a'$ is sampled from the current policy $\pi_\theta$ with target policy smoothing.

The critic loss is the Mean Squared Error:
$$L(\phi_i) = \mathbb{E}_{(s, a, y) \sim \mathcal{D}} \left[ (Q_{\phi_i}(s, a) - y)^2 \right]$$

## 3. Regularization Mechanisms

### KL Divergence Penalty
The KL penalty $D_{KL}(\pi_\theta \| \pi_{ref})$ prevents the policy from deviating too far from the reference policy in a single update. This is especially important in off-policy learning where the data in the buffer might be stale.
For Gaussian policies $\mathcal{N}(\mu_0, \sigma_0)$ and $\mathcal{N}(\mu_1, \sigma_1)$, the KL divergence is:
$$D_{KL} = \sum \left( \log \frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2} - \frac{1}{2} \right)$$

### Target Policy Smoothing
During critic updates, we add small noise to the target action:
$$a' = \pi_{target}(s_{t+n}) + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \tilde{\sigma}), -c, c)$$
This smooths the value surface and prevents the policy from exploiting narrow peaks in the Q-function.
