---
title: "Measure-Theoretic View of Policy Gradients"
draft: false
date: 2025-02-22
---

# Introduction

## Why a Measure-Theoretic View of Policy Gradients?

Reinforcement learning (RL) has ~~long~~ always relied on probability densities and likelihood ratios to compute policy gradients. The standard derivation comes to this conclusion:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E} \left[ R \nabla_\theta \log \pi_\theta(a | s) \right]
$$

where $J(\pi_\theta)$ is the objective function (e.g. expected reward), $\pi_\theta$ is the policy, $R$ is the reward, and $\nabla_\theta \log \pi_\theta(a | s)$ is the gradient of the log policy. Basically what we covered previously.

This formulation, while widely used, assumes that policies have well-defined probability density functions. However, this assumption breaks down in several cases:
- Continuous action spaces: Not all policies over continuous actions admit densities (e.g. Gaussian policies, almost all of Mujoco envs)
- Entropy-regularized RL: Some regularization schemes (entropy based) implicitly modify policies in ways that make densities hard to define
- Optimal transport and Wasserstein-based methods: Probability mass shifts are more naturally expressed in terms of measures rather than densities (This is just to express RL in a new light, to draw more inspiration from other fields)

~~but ultimately, I believe almost all of DL is just optimal transport~~

By moving to a measure-theoretic view, we unlock a more general perspective on policy optimization — one that applies even when probability densities do not exist or are inconvenient to work with. This shift is not just a theoretical exercise, but has practical implications for algorithm design (mainly), stability, and ~~efficiency~~. By shifting this view, we can explore RL in a broader way and just see on whether we can apply some tips and tricks

In this blog, I will try to explain the measure-theoretic view of policy gradients in a way that is accessible to RL practitioners. I will introduce key concepts as needed, and provide references to more detailed resources for those who want to delve deeper, but this blog is not written by a mathematician and I do not claim to be one, so any FEEDBACK or CRITICISM is welcome, but please be gentle I have a fragile ego and OCD.

I assume familiarity with:
- Basic reinforcement learning (Markov decision processes, policy gradients)
- Probability theory (expectations, probability distributions)
- Calculus and linear algebra

But even if you don't have any of these, I hope you can still enjoy the blog and learn something new.

I do not assume prior knowledge of measure theory (in fact, I assume and hope at the opposite), and I will try to introduce key concepts as needed.

## What are we gonna do in this blog?

1. Measure-Theoretic Setup for Policies
- A brief primer on measure theory (I promise it's not as scary as it sounds)
- How policies can be viewed as probability measures rather than functions (Trust me, it is worth it)
- The concept of occupancy measures, which describe how policies interact with the environment (Not simple, but effective drop-in replacement for trajectories)
2. Policy Gradients via the Radon-Nikodym Derivative
- The limitations of traditional policy gradient methods
- How the Radon-Nikodym derivative provides a more general formulation
- What this means for RL optimization and algorithm design
3. Policy Optimization and Convex Analysis
- Reformulating RL objectives using integrals over measures
- How this leads to convex formulations of policy optimization
- The role of regularization and KL-divergence in measure space

# Measure-Theoretic Setup for Policies

## A brief primer on measure theory

Reinforcement learning (RL) is a probabilistic framework, but it is often presented using probability densities — functions like  $\pi(a|s)$  that describe how an agent selects actions, we just "predict" actions giving the state, meaning that we are likely to take action $a$ given state $s$. This is a convenient, but restrictive viewpoint, because not all policies are that beautiful and nice. Policies can be deterministic, discrete-continuous hybrid, etc. So, we gotta have a more general view of policies

So, the motivation is to generalize policy and its gradients beyond explicit density assumptions and even step down from the probabilistic approach and <b>ascend</b> to the measure theoretic view

Before we define policies in this new way, we need to establish some fundamental concepts in measure theory, ones who are familiar with measure theory can skip this section (and entire blog, if you are toxic).

### Sigma-Algebras

A sigma-algebra ($\sigma$-algebra) is formally just a set with some properties. Formally, it satisfies three properties:
1. Contains the entire space: If we’re measuring actions in  $\mathcal{A}$ , then the full space  $\mathcal{A}$  must be included
2. Closed under complements: If a subset  $A$  is in it, so is its complement  $A^c$ 
3. Closed under countable unions: If  $A_1, A_2, \dots$  are in it, then their union is also in it

Some small example:

- Borel $\sigma$-algebra:
    - The smallest $\sigma$-algebra containing all open sets in a space
    - Ensures we can assign probabilities to intervals, discrete actions, and mixtures

### Measures

A measure assigns a “size” (or probability, in a more intuitive and example sense) to subsets of a space in a consistent way. A measure $\mu$ on a $\sigma$-algebra $\mathcal{F}$ is defined as a function:

$$
\mu: \mathcal{F} \to [0, \infty]
$$

that satisfies:
1. The empty set has measure zero: $\mu(\emptyset) = 0$
2. Countable additivity: If $A_1, A_2, \dots$ are disjoint measurable sets, then:

$$
\mu\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} \mu(A_i).
$$

Intuitively, one measure is just length or area or volume, etc. It just shows how to "measure" the size of a set

A measure is a <b>probability measure</b> if $\mu(\mathcal{A}) = 1$, ensuring that probabilities sum to one

### The Radon-Nikodym Derivative

The Radon-Nikodym derivative generalizes the concept of probability densities. Given two measures $\mu$ and $\nu$, if $\mu$ is absolutely continuous with respect to $\nu$ (denoted $\mu \ll \nu$), there exists a function $f$ such that:

$$
\mu(A) = \int_A f d\nu
$$

for all measurable sets $A$. This function $f$ is called the Radon-Nikodym derivative, written as:

$$
\frac{d\mu}{d\nu}
$$

If you dont get it now, dont worry you will get it later. (I always dreamt of writing this sentence in a blog. Thank you, Andrew Ng senpai)

## How policies can be viewed as probability measures rather than functions

A probability measure $\pi$ is a function that "assigns probabilities" to subsets of the action space. Instead of defining a density function $\pi(a | s)$, we define $\pi$ as a probability measure (be cautious as I use the same notation for both the policy and the probability measure):

$$
\pi(s, A) = \int_A \pi(s, da)
$$

where:
- $A$ is a measurable subset of the action space $\mathcal{A}$
- $\pi(s, A)$ represents the probability of selecting an action within $A$ given state $s$

I can already sense a question: "Why the hell are you doing this? What is the benefit of this?"

Well, as we have discussed previously, this is to generalize the policy and its gradients beyond density. We will use it later to reformulate the policy gradient theorem in a more general way. (but overall I get that this blog feels like an overkill to an understanding, but in general its just a way to generalize the mathematical view)

We will also note about Markov kernel, which is also related to the policy, as a matter of fact, policy is a Markov kernel:

$$
\pi: S \times \mathcal{F}(\mathcal{A}) \to [0,1]
$$

where:
- $S$ is the state space.
- $\mathcal{F}(\mathcal{A})$ is the $\sigma$-algebra over the action space.
- $\pi(s, A)$ gives the probability of selecting an action in $A$ given state $s$

Why? Just to list all the properties of the policy in a more rigorous (I know how much math guys love this word) way.

## The concept of occupancy measures, which describe how policies interact with the environment

In RL, especially in policy gradients, we frequently encounter formulation of a trajectory (well because we actually need to converge to an optimal policy). Instead of tracking individual trajectories, we often need a global measure of how often a policy visits different state-action pairs, this can be as expressive as just a bunch of collected trajectories. This is where occupancy measures come in

An occupancy measure provides a probability distribution over state-action pairs under a given policy. Instead of working with sampled transitions, we define a stationary distribution that captures long-term visitation frequencies. This concept allows us to reformulate quite a lot of objectives in RL in a more general way using not some Monte Carlo sampled trajectories, but some precise visitation frequencies

# Policy Gradients via the Radon-Nikodym Derivative

## Derivation of policy gradients using the Radon-Nikodym derivative

We will start with a recap of the policy gradient theorem, as it is the foundation of the policy gradient methods.

The goal in RL is to maximize the expected return:

{{<rawhtml>}}

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)]
$$

{{</rawhtml>}}

where:
- $\tau = (s_0, a_0, s_1, a_1, \dots)$ is a trajectory
- $R(\tau)$ is the return (e.g. cumulative (discounted or not) reward)
- $\pi_\theta$ is the policy parameterized by $\theta$

Using the gradient of an expectation, we try to find the gradient of the objective:

{{<rawhtml>}}

$$
\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)]
$$

{{</rawhtml>}}

from where we can derive the gradient of the objective:

{{<rawhtml>}}

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) R(\tau) d\tau.
$$

{{</rawhtml>}}

where $p_\theta(\tau)$ is the probability density of trajectory $\tau$ under the policy $\pi_\theta$ obviously

Now we use the identity, which is just a log derivative:

{{<rawhtml>}}

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau),
$$

{{</rawhtml>}}

which gives:

{{<rawhtml>}}

$$
\nabla_\theta J(\theta) = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) R(\tau) d\tau.
$$

{{</rawhtml>}}

Rewriting as an expectation:

{{<rawhtml>}}

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right].
$$

{{</rawhtml>}}

This is the classic policy gradient theorem, then we can rewrite simple $p_\theta(\tau)$ with policy and thats basically it

### So what for do we need Radon-Nikodym derivative?

Well, we missed it. Lets CIRCLE BACK to log derivative trick and see something new there

The log-derivative trick:

{{<rawhtml>}}

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
$$

{{</rawhtml>}}


In fact, it can be seen as a special case of the Radon-Nikodym derivative when considering an infinitesimally small perturbation of $\theta$ (remember we are trying to change the policy from some base $\theta_0$ to some optimal $\theta{\prime}$). Essentially, the policy gradient theorem can be thought of as an application of the Radon-Nikodym derivative, where instead of changing the entire measure, we take the derivative of the density function directly, but first some preliminary.

This is why the policy gradient theorem can be rewritten as:

{{<rawhtml>}}

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \mathbb{P}_\theta} \left[ \nabla_\theta \log \frac{d\mathbb{P}_\theta}{d\mathbb{P}_{\theta{\prime}}} R(\tau) \right]
$$

{{</rawhtml>}}

I know that this feels out of the blue, but lets just recap quickly the intuition of Radon-Nikodym derivative:

It shows us how one measure (in our case $\mathbb{P}\theta$) changes with respect to another measure (in our case $\mathbb{P}{\theta{\prime}}$), (sorry for notation, I somehow cannot fix MathJax here and ChatGPT cant help either, but you understand its underscript). Formally it looks something like this:

{{<rawhtml>}}

$$
d\mathbb{P}_\theta = p(\tau) d\mathbb{P}_{\theta{\prime}}(\tau)
$$

{{</rawhtml>}}

Looks better now? I guess, so! but still what it means and why? Now what? Now, that we now that it is just a special case of Radon-Nikodym theorem we can retract the assumptions under which it works (hint: it works only with infinitesmall change of $\theta$ - differential), but are we on practice having this infinitesmall changes? The answer is NO! that is why ~~my glorious and precious king~~Schulman et al. and Sham Kakade (and other guys not to be offensive) invented upgrades to Policy Gradients such as TRPO (go where its infinitesmall enough), Natural Gradients, and so on. Even if we look at modern PPO clipped objective we can see that it resembles our Radon-Nikodyme derivative quite closely.

OK, enough with yapping, but we still can arrive back at our policy gradient using just the assumption and we are back at it:

{{<rawhtml>}}

$$
\nabla_\theta J(\theta) = \mathbb{E}{\tau \sim \pi{\theta}} \left[ \nabla_\theta \log p_\theta(\tau) R(\tau) \right]
$$

{{</rawhtml>}}

## Occupancy measures to the fight!

Instead of reasoning directly over trajectory distributions  $p_\theta(\tau)$, we can shift our perspective to state-action occupancy measures (yes, the ones introduced before). Lets write it formally:

{{<rawhtml>}}

$$
d^\pi(s, a) = \sum_{t=0}^{\infty} \gamma^t P(s_t=s, a_t=a | \pi)
$$

{{</rawhtml>}}


where $P(s_t=s, a_t=a | \pi)$  is the probability of visiting  $(s, a)$  at time $t$ under policy $\pi$. This occupancy measure defines a stationary distribution over state-action pairs (yeah, as always). Since we are now working only with occupancy measures, we will try to drop in the replacement in our policy gradient theorem:

First, we will unroll what is $p_\theta(\tau)$ actually is:

{{<rawhtml>}}
$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) P(s_{t+1} | s_t, a_t)
$$
{{</rawhtml>}}

By doing this, we can actually find this relation to be true:

{{<rawhtml>}}
$$
\mathbb{E}_{\tau \sim p\theta} \left[ \sum_{t=0}^{T} f(s_t, a_t) \right] = \sum_{t=0}^{T} \sum_{s, a} P(s_t = s, a_t = a | \pi) f(s, a) = \sum_{s, a} d^\pi(s, a) f(s, a)
$$
{{</rawhtml>}}

You probably are asking now, why sums over $f(s_t, a_t)$ well, that is because these are our sampled trajectories, we just unrolled them

# Policy Optimization and Convex Analysis

Now that we have discovered (and used!) some of the ~~most brutal and vicious and most ruthless~~ more broad formalism, we can actually broaden up other things in RL right?

## Reformulating RL Objectives Using Integrals Over Measures

The standard RL objective is ~~get the most bitches~~ get the most discounted cumulative reward (return):

{{<rawhtml>}}
$$
J(\pi) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right]
$$
{{</rawhtml>}}

Instead of summing over trajectories, we rewrite it using occupancy measures:

{{<rawhtml>}}
$$
J(\pi) = \int_{\mathcal{S} \times \mathcal{A}} d^\pi(s, a) r(s, a)
$$
{{</rawhtml>}}

see, its easy? If we read this integral in a natural way we would say "sum up all the occupied state-action pairs weighted by the reward of state-action"

This is important for many things:
- This allows for us to use some variational magic here (hello bayesian RL)
- occupancy measure is guaranteed that it is consistent with Markov transition dynamics
- RL becomes just a convex optimization linear program problem (only tabular ofc, but still something!)

## Why Is Convexity Useful?

As mentioned before, RL can just become a convex optimization problem which eliminates a lot of troubles. Lets take an example here:

Instead of optimizing over a non-convex space of policies $\pi$, we optimize over the convex set of occupancy measures:

{{<rawhtml>}}
$$
\max_{d^\pi} \int_{\mathcal{S} \times \mathcal{A}} d^\pi(s, a) r(s, a)
$$
{{</rawhtml>}}

Of course with respect to:
1. Stationarity - other way round occupancy measure breaks
2. Normalization - other way round probability breaks

and this opens a brand new world for gradient methods and we can just optimize over measures, it is explicitly used in TRPO written by ~~my glorious king~~ Schulman et al.

## Regularization and KL

Regularization in measure theoretic interpretation becomes way more justified and even kinda intuitive rather then just a pure hack. We will cover a case of KL divergence as it is frequently used as a REGULARIZER in RL

{{<rawhtml>}}

$$
D_{\text{KL}}(\pi || \pi_0) = \int_{\mathcal{S} \times \mathcal{A}} d^\pi(s, a) \log \frac{d^\pi(s, a)}{d^{\pi_0}(s, a)}
$$

{{</rawhtml>}}

(dont worry this is just KL written in our already familiar terms)

What we are doing here? a lot of things actually:
- we balance explore-exploit problem - by addressing closeness to prior policy
- we encourage smooth updates - by preventing HUGE steps in divergence
- ~~we encourage capitalism~~ We dont

# Conclusion

Through a measure-theoretic perspective, we have uncovered a broader and more principled way to understand policy gradients, moving beyond the limitations of density-based formulations. By viewing policies as probability measures and leveraging tools like the Radon-Nikodym derivative, we gain a more flexible framework that naturally extends to cases where densities may not be well-defined. (yeah it was written by ChatGPT)

By actually examining how Radon-Nikodym derivative and occupancy measures are used we can leverage more INSIGHTS into interesting directions in RL. We can see RL from another angle. ~~Our skin becomes clearer~~. We actually can gain theoretical information on why something works and something is not without just relying on set of hacks (its just beautiful, but I still am dumb enough to actually derive it)

In essence, measure-theoretic RL is not just a mathematical abstraction — it has direct practical implications, that already influenced algorithm design (as examples we have taken TRPO, natural gradients and KL-regularized methods) and improving stability in policy optimization. By ~~ascending~~ stepping beyond conventional probability densities, we allow ourselves to see reinforcement learning through a different lens — one that may ultimately lead to more robust, ~~efficient,~~ (yeah calculate hessian or fisher matrix, I will wait) and theoretically grounded methods