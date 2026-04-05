## E-Step

In the E-step, MPO computes an auxiliary distribution over actions by
reweighting the current policy with critic values. The goal is to improve the
action distribution while keeping it close to the current policy  
$\pi_i(a \mid s)$.

In the non-parametric variant of MPO, this is done by solving a constrained
optimization problem: maximize the auxiliar distribution $q(a \mid s)$,
subject to a KL constraint and normalization.

---

### Constrained Optimization Problem

$$
\begin{aligned}
\max_q \quad &
\int \mu(s)\int q(a \mid s)\,Q^{\pi_i}(s,a)\,da\,ds \\
\text{subject to} \quad &
\int \mu(s)\,\mathrm{KL}\!\left(q(a \mid s)\,\|\,\pi_i(a \mid s)\right)\,ds
< \epsilon, \\
&
\int \int \mu(s)\ q(a \mid s)\,dads = 1
\end{aligned}
$$

Here, $\mu(s)$ denotes the state distribution approximated by replaybuffer samples. The KL constraint ensures that the
updated distribution remains close to the current policy. 

---

### Lagrangian Formulation

To solve this constrained optimization problem, we form the Lagrangian

$$
\begin{aligned}
\mathcal{L}(q,\eta,\gamma)
=\;&
\int \mu(s)\int q(a \mid s)\,Q^{\pi_i}(s,a)\,da\,ds \\
&+
\eta\left(
    \epsilon
    -
    \int \mu(s)\,\mathrm{KL}\!\left(q(a \mid s)\,\|\,\pi_i(a \mid s)\right)\,ds
\right) \\
&+
\gamma ( 1 -  \int\int \mu(s) q(a \mid s)\,dads)
\end{aligned}
$$

where $\eta \ge 0$ is the Lagrange multiplier for the KL constraint and
$\gamma(s)$ enforces normalization.

Taking derivatives w.r.t. $q(a \mid s)$ and rearranging termns yields the closed solution of the auxiliar distribution and dual function

$$
q_i(a \mid s) \propto \pi(a \mid s, \theta_i)\,
\exp\!\left(
\frac{Q_{\theta_i}(s,a)}{\eta^*}
\right)
$$

$$
g(\eta)
=
\eta \epsilon
+
\eta \int \mu(s)\,
\log \int
\pi(a \mid s, \theta_i)\,
\exp\!\left(
\frac{Q_{\theta_i}(s,a)}{\eta}
\right)
\, da \, ds
$$

---

### Practical Approximation (Softmax over Samples)

In practice, the integral is intractable and approximated via sampling from the replaybuffer.

For each state sample we sample $N$ actions $a_i \sim \pi(\cdot \mid s)$ and compute:

$$
q_{ij} = q(a_i, s_j)
=
\frac{
    \exp\!\left(\frac{Q(s_j, a_i)}{\eta}\right)
}{
    \sum_{l=1}^{N}
    \exp\!\left(\frac{Q(s_j, a_l)}{\eta}\right)
}.
$$

This corresponds to a softmax over sampled Q-values and is the form used in
practical implementations as proposed in [cite 2].

The dual function can be mimimized in practice by a few steps of gradient descent by the folling equation:

$$
\eta
=
\arg\min_{\eta}
\left(
\eta \epsilon
+
\eta \sum_{j=1}^{K}
\frac{1}{K}
\log
\left(
\sum_{i=1}^{N}
\frac{1}{N}
\exp\!\left(
\frac{Q(s_j, a_i)}{\eta}
\right)
\right)
\right)
$$

## Action Penalty

In continuous control, sampling from our policy $\pi(a \mid s)$ can produce out-of-bound actions, for which the critic may assign unreliable values because those samples were never observed during training. This can destabilize learning. To mitigate this issue, we include an additional out-of-bounds action penalty in the E-step, following the multi-objective MPO implementation by Google DeepMind \cite{acme_mpo_py}. 

First, we compute the negative quadratic distance of the action and the action bound:
$$
\Delta(a) = a -\mathrm{clip}(a, a_{\min}, a_{\max})
$$

$$
c(a) = - \lVert \Delta(a) \rVert^2
$$

After that, we use the formula of the original E-step and compute a new auxiliar distribution for the action penalty $q_{\text{pen}}(a \mid s)$:

$$
q_{\text{pen}}(a \mid s)
=
\frac{
\exp\left(\frac{c(a)}{\eta_{\text{pen}}}\right)
}{
\sum_{i}
\exp\left(\frac{c(a_i)}{\eta_{\text{pen}}}\right)
}
$$

The term "action penalty" is somewhat misleading, since we do not add an explicit penalty term to the reward or the critic objective. Instead, we mix two sets of E-step weights. We combine both weightings via a mixing factor $\lambda$ and renormalize the result, yielding the final auxiliary distribution $q(a\mid s)$.

$$
q(a \mid s)
=
(1 - \lambda)\, q_{\text{MPO}}(a \mid s)
+
\lambda\, q_{\text{pen}}(a \mid s)
$$

$$
q(a \mid s)
\leftarrow
\frac{
q(a \mid s)
}{
\sum_i q(a_i \mid s)
}
$$

## Results of Action Penalty

[place holder figure results]

From the learning curves in, we observe that both tanh-squashing of the policy mean and the out-of-bounds action penalty improve early training, leading to faster return increases and improved sample efficiency. The best-performing configuration combines tanh on the mean with the action penalty, while the second-best variant uses the action penalty without tanh as activation function. This indicates that incorporating the penalty term in the E-step stabilizes learning and accelerates early progress. After roughly one million environment steps, however, all variants converge to a very similar performance level. A plausible explanation is that the penalty has the strongest impact early on, when policy variance is high and out-of-bound samples occur more frequently. As training progresses, the policy becomes increasingly deterministic, making such samples rare and reducing the influence of the penalty term.