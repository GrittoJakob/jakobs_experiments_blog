## M-Step

In the M-step, the policy is updated to match the auxiliary distribution while remaining close to the target policy to ensure stability. This is achieved by maximizing a likelihood objective under a KL divergence constraint, resulting in a constrained optimization problem.

This constrained problem can be written in Lagrangian form as:

$$
\max_{\pi}
\;
\int \mu_q(s)
\int q(a \mid s)\,\log \pi(a \mid s, \theta_i)\,da\,ds
\;-\;
\lambda
\int \mu_q(s)\,
\mathrm{KL}\!\left(
\pi(a \mid s, \theta_i)\,\|\,\pi_{\text{target}}(a \mid s, \theta)
\right)
ds
$$

The original paper proposes a decoupled version of the M-step, where the KL divergence is separated into a contribution of the mean and the covariance of the policy. This leads to the following objective:


$$
L(\theta, \eta_\mu, \eta_\Sigma)
=
\int \mu_q(s)
\int q(a \mid s)\,\log \pi(a \mid s, \theta_i)\,da\,ds
+
\eta_\mu (\epsilon_\mu - C_\mu)
+
\eta_\Sigma (\epsilon_\Sigma - C_\Sigma)
$$
with 


$$
C_\Sigma
=
\int \mu_q(s)\;
\frac{1}{2}
\left(
\mathrm{tr}\!\left(\Sigma^{-1}\Sigma_i\right)
- n
+ \ln \frac{\Sigma}{\Sigma_i}
\right)
\, ds
$$

$$
C_\mu
=
\int \mu_q(s)\;
\frac{1}{2}
(\mu - \mu_i)^\top \Sigma^{-1} (\mu - \mu_i)
\, ds
$$

Unlike the E-step, the Lagrangian does not admit a closed-form solution. Therefore, we solve this constrained optimization problem by alternating min-max optimization where we first fix our lagrangian multipliers $\eta$ and optimise for $\theta$ and then fix $\theta$ and solve vor $\eta$.


