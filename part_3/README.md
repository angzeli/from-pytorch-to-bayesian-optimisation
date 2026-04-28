# 📘 Part 3 — Modelling Unknown Functions and Bayesian Optimisation Foundations

Part 3 is the conceptual bridge from optimisation dynamics to **Bayesian Optimisation**.

In Part 2, the objective was treated as something we could inspect, analyse, or evaluate directly.
In Part 3, that assumption is relaxed.

We now study what happens when the objective function is:

- expensive to evaluate,
- only partially observed,
- and better handled through a learned surrogate than through brute-force search.

This part introduces the core ideas needed before practical Bayesian Optimisation workflows can be understood:

- why expensive objectives require modelling,
- how surrogate models approximate unknown functions,
- why prediction alone is not enough without uncertainty,
- Gaussian Processes as principled probabilistic surrogates,
- and acquisition functions for deciding where to evaluate next.

The goal is not just to learn *how* Bayesian Optimisation works mechanically, but to understand **why modelling and uncertainty are essential once the objective is no longer freely available**.

---

## 🎯 Scope and Philosophy

Across these tutorials and workshops, surrogate modelling is treated as:

- a response to limited objective access, not just curve fitting,
- a probabilistic reasoning problem, not just interpolation,
- and a decision-support framework, not just a prediction task.

The material emphasises:

- the need for surrogate models in expensive optimisation,
- the difference between prediction and confidence,
- uncertainty-aware reasoning about where data are informative,
- Gaussian Processes as distributions over functions,
- and acquisition functions as the decision layer that turns a surrogate into an optimisation strategy.

By the end of Part 3, you should be comfortable answering questions like:

- *Why is a mean prediction alone not enough in expensive optimisation?*
- *What does a Gaussian Process actually represent?*
- *How do observations update both the GP mean and uncertainty?*
- *Why does Bayesian Optimisation need acquisition functions rather than just a surrogate model?*

---

## 📚 Contents

### Tutorial 1 — Why Model an Unknown Objective?
**Focus:** Motivation for surrogate modelling in expensive optimisation.

- When direct optimisation becomes impractical
- Sparse observations and limited evaluation budgets
- Surrogate models as stand-ins for expensive objectives
- Why black-box optimisation needs modelling
- From direct descent to model-guided search

📓 `tutorial_01_why_model_an_unknown_objective.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 2 — Prediction, Uncertainty, and Confidence
**Focus:** Why prediction alone is not enough.

- Mean prediction vs predictive confidence
- Interpolation vs extrapolation
- Heuristic uncertainty bands
- Uncertainty as a function of data support
- Exploration–exploitation intuition before probabilistic modelling

📓 `tutorial_02_prediction_uncertainty_and_confidence.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 3 — Gaussian Processes as Surrogate Models
**Focus:** A principled probabilistic surrogate framework.

- Kernels as covariance functions
- GP priors and posterior updates
- Prior samples vs posterior samples
- Posterior mean and posterior uncertainty
- Kernel hyperparameters such as lengthscale and variance
- Sequential updating of the GP surrogate

📓 `tutorial_03_gaussian_processes_as_surrogate_models.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 4 — Choosing Where to Evaluate Next
**Focus:** The decision layer of Bayesian Optimisation.

- Why minimising the posterior mean alone is too greedy
- Acquisition functions as next-point selection rules
- Lower Confidence Bound (LCB)
- Probability of Improvement (PI)
- Expected Improvement (EI)
- The full sequential BO loop in one dimension

📓 `tutorial_04_choosing_where_to_evaluate_next.ipynb`  
🛠 Worked version in `worked/`

---

### Workshop 1 — Visualising Surrogate Behaviour
**Focus:** Building intuition for model-guided reasoning.

- How surrogate mean and uncertainty move with data
- Where uncertainty contracts and where it persists
- Local vs global support in surrogate fitting
- Interpreting surrogate failures and limitations
- Why probabilistic surrogates are useful even when imperfect

📓 `workshop_01_visualising_surrogate_behaviour.ipynb`  
🛠 Worked version in `worked/`

---

### Workshop 2 — From Surrogates to Sequential Optimisation
**Focus:** Connecting GP modelling to Bayesian Optimisation behaviour.

- Acquisition-driven search as sequential decision-making
- Why BO does not improve the best value at every step
- Model improvement vs objective improvement
- Surrogate accuracy vs optimisation usefulness
- Interpreting BO as model-guided information gathering

📓 `workshop_02_from_surrogates_to_sequential_optimisation.ipynb`  
🛠 Worked version in `worked/`

---

## 🧠 How Part 3 Fits into the Series

Part 3 is **probabilistic and model-based**.

It marks the point where optimisation is no longer treated as a process acting directly on a fully accessible objective.
Instead, optimisation becomes increasingly about:

- representing what is known,
- quantifying what is uncertain,
- and deciding what information is most valuable to gather next.

So the series now shifts:

- from gradients and optimisation dynamics,
- to surrogate modelling and uncertainty-aware search.

This part intentionally develops the ideas from the ground up before introducing heavy tooling.

**Part 4** will take these foundations and turn them into practical workflows using **BoTorch**, where Gaussian Process surrogates and acquisition functions are implemented in a modern Bayesian Optimisation library.

---

## ✅ Prerequisites

- Completion of **Part 2**
- Basic familiarity with optimisation trajectories and objective functions
- Some comfort with probability, covariance, and Gaussian distributions
- Curiosity about *how to optimise when the objective itself is only partially known*

---

**Author:** Angze Li   
**Version:** v1.0
