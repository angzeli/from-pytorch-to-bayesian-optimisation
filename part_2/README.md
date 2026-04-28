# 📘 Part 2 — Optimisation Dynamics: Geometry, Conditioning, Momentum, and Non-Convexity

Part 2 studies **how optimisation behaves as a dynamical process once gradients are applied repeatedly over time**.

In Part 1, gradients were treated as local sensitivity objects.
In Part 2, those gradients become update directions, and the central question becomes:

> **what kinds of trajectories do repeated gradient-based updates produce?**

Rather than presenting gradient descent as a black-box optimisation recipe, this part treats optimisation as a structured dynamical system shaped by:

- the objective geometry,
- the learning rate,
- the conditioning of the problem,
- the presence or absence of momentum,
- and the distinction between convex and non-convex landscapes.

The goal is not just to learn *how* to run optimisation, but to understand **why different optimisation behaviours appear**:
- slow convergence,
- oscillation,
- zig-zagging,
- instability,
- sensitivity to initialisation,
- and the constructive or destructive role of noise.

---

## 🎯 Scope and Philosophy

Across these tutorials, optimisation is treated as:

- a **discrete-time dynamical system**, not just an update rule,
- a **geometric process**, not just numerical minimisation,
- and a behaviour determined jointly by gradients, curvature, conditioning, inertia, and landscape structure.

The material emphasises:

- repeated updates and trajectory analysis,
- linear dynamical interpretations of gradient descent,
- the geometry of 2D loss surfaces,
- conditioning and reparameterisation,
- momentum as accumulated velocity,
- and optimisation behaviour in non-convex and noisy settings.

By the end of Part 2, you should be comfortable answering questions like:

- *When is gradient descent stable, oscillatory, or divergent?*
- *Why does ill-conditioning produce zig-zagging and slow progress?*
- *How does momentum change the optimisation dynamics?*
- *Why can non-convex optimisation depend strongly on initialisation?*
- *When does noise hurt convergence, and when can it actually help?*

---

## 📚 Contents

### Tutorial 1 — Gradient Descent as Dynamical System
**Focus:** Treating gradient descent itself as a discrete-time dynamical system.

- Single gradient descent steps and repeated updates
- 1D quadratic optimisation as a linear dynamical system
- Stability as a function of learning rate
- 2D trajectories, loss landscapes, and vector fields
- Basin structure in a 1D double-well
- Empirical stability across learning rates
- Loss decay under ill-conditioning

📓 `tutorial_01_gradient_descent_as_dynamical_system.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 2 — Geometry and Conditioning of Optimisation
**Focus:** How curvature and conditioning shape optimisation trajectories.

- Contours of quadratic loss surfaces
- Isotropic vs ill-conditioned quadratics
- Condition number as a quantitative measure of difficulty
- How conditioning affects convergence speed
- Reparameterisation and scaling
- When coordinate changes help — and when they make things worse

📓 `tutorial_02_geometry_and_conditioning_of_optimisation.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 3 — Momentum as a Dynamical System
**Focus:** Understanding momentum as structured optimisation dynamics rather than a heuristic trick.

- Momentum updates as gradient accumulation plus inertia
- Comparison between gradient descent and momentum
- Loss-vs-step behaviour
- Steep vs flat directions under momentum
- Sweeping the momentum parameter `β`
- Stability as a function of learning rate
- When momentum helps and when it hurts
- Position–velocity interpretation of momentum dynamics

📓 `tutorial_03_momentum_as_a_dynamical_system.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 4 — Optimisation Beyond Convexity
**Focus:** What changes once the landscape is non-convex and optimisation is no longer deterministic in effect.

- A 2D multi-well non-convex landscape
- Sensitivity to initialisation
- Noisy gradients on convex objectives
- Noise and convergence speed
- Random restarts and multiple minima
- Noise-assisted basin escape
- When noise helps exploration and when it hurts local convergence
- Final loss under fixed compute budgets
- Trade-offs between noise level and optimisation quality

📓 `tutorial_04_optimisaiton_beyond_convexity.ipynb`  
🛠 Worked version in `worked/`

---

## 🧠 How Part 2 Fits into the Series

Part 2 is **dynamic, geometric, and behavioural**.

It takes the local gradient concepts of Part 1 and asks what happens when those gradients are used repeatedly in actual optimisation procedures.

So the emphasis shifts:

- from **what gradients are**,
- to **what gradients do over time**.

This part is still relatively low-dimensional and visual.
That is intentional.

The goal is to build deep intuition for optimisation trajectories before moving to settings where the objective is no longer directly accessible.

**Part 3** will relax one of the strongest assumptions used here:
that the objective can be inspected and evaluated directly whenever we want.

Instead, it will study what happens when the objective is:

- expensive,
- only partially observed,
- and better approached through a learned surrogate model.

That is where the series transitions from optimisation dynamics to **modelling unknown functions** and then to **Bayesian Optimisation**.

---

## ✅ Prerequisites

- Completion of **Part 1**
- Basic multivariable calculus and linear algebra
- Familiarity with gradients and computation graphs
- Curiosity about *why optimisation behaves the way it does over repeated updates*

---

**Author:** Angze Li 
**Version:** v1.0
