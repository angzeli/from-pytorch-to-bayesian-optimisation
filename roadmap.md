# 🧭 Roadmap: Part 3–5 and Beyond

> **This section outlines the roadmap for the rest of this repository.**  
> It provides a high-level view of what will be covered next, how the pieces connect, and where the project is heading.

---

## 🎯 Overall Direction

So far, we have studied **optimisation as a dynamical system** and explored how **geometry and conditioning** affect its behaviour.

However, a key limitation has emerged:

> **Gradient-based optimisation alone is not sufficient when the objective is complex, noisy, or expensive to evaluate.**

The next parts of this repository aim to address this limitation by introducing:
- **modelling of unknown functions**, and  
- **decision-making under uncertainty**,  

which together lead to **Bayesian Optimisation (BO)**.

---

# 📘 Part 3 — Modelling Unknown Functions (Bridge)

> *Goal: Introduce surrogate models and uncertainty as a foundation for Bayesian Optimisation.*

This part serves as a **transition** from optimisation (Part 2) to BO.

Rather than directly minimising a function, we now:
> **build a model of the function and use it to guide optimisation.**

---

### Topics covered

- Why modelling is needed (expensive / noisy objectives)
- Surrogate models and data-efficient optimisation
- Mean vs uncertainty (variance)
- Confidence and prediction under limited data
- Gaussian Processes (GPs) — intuition and basic mechanics

---

### Structure (planned)

- **Tutorial 1** — Why modelling expensive unknown functions?
- **Tutorial 2** — Prediction is not enough: uncertainty and confidence
- **Tutorial 3** — Gaussian Processes as surrogate models (intuition-focused)

---

### Key takeaway

> **We can replace direct optimisation with model-guided decision-making.**

---

# 📘 Part 4 — Bayesian Optimisation (Core Focus)

> *Goal: Learn how to optimise expensive, unknown functions using probabilistic models.*

This is the **main focus** of the repository.

Bayesian Optimisation combines:
- a **probabilistic model** (from Part 3), and  
- an **optimisation strategy** (from Part 2),  

to decide **where to evaluate next**.

---

### Core loop

data → model → acquisition → optimise → new data → repeat

---

### Topics covered

- Exploration vs exploitation
- Acquisition functions (EI, UCB, PI)
- Optimising acquisition functions
- Multi-start strategies and local optimisation issues
- Practical BO implementation (BoTorch)

---

### Structure (planned)

- **Tutorial 1** — BO intuition (exploration vs exploitation)
- **Tutorial 2** — GP + BO loop
- **Tutorial 3** — Acquisition optimisation
- **Tutorial 4** — Practical BO (BoTorch)
- **Tutorial 5** — Advanced topics (optional)

---

### Key takeaway

> **Optimisation becomes a sequential decision-making problem under uncertainty.**

---

# 📘 Part 5 — Advanced Bayesian Optimisation (Optional)

> *Goal: Extend BO to more realistic and challenging settings.*

This part explores extensions used in research and real-world applications.

---

### Possible topics

- Batch Bayesian Optimisation
- Constraints in BO
- High-dimensional optimisation
- Noisy and stochastic objectives
- Advanced acquisition strategies

---

### Note

This part is **optional** and may be developed based on:
- available time, and  
- research relevance.

---

# 🚀 Beyond BO

After completing Bayesian Optimisation, there are several possible directions:

---

## Option A — Applied Machine Learning (PyTorch)

- Datasets and data pipelines  
- Training loops and evaluation  
- Model architectures  

---

## Option B — Scientific Visualisation (matplotlib)

- Publication-quality figures  
- Advanced plotting techniques  
- Data analysis workflows  

---

## Option C — Research Projects

- Applying BO to real scientific problems  
- Extending BO methods  
- Building reusable optimisation tools  

---

# 🧠 Big Picture

This repository is structured around a central progression:
Part 2: Introduction             → How we work
Part 2: Optimisation             → How we move
Part 3: Modelling               → How we represent uncertainty
Part 4: Bayesian Optimisation   → How we decide

---

## Final perspective

> **Optimisation alone is not enough.**  
> **Modelling alone is not enough.**  

It is their combination that enables:
> **efficient decision-making in complex, real-world problems.**
