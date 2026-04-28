# 📘 Part 4 — Practical Bayesian Optimisation Workflows in BoTorch

Part 4 takes the conceptual foundations of Part 3 and turns them into **practical Bayesian Optimisation workflows in BoTorch**.

In Part 3, the emphasis was on understanding *why* Bayesian Optimisation needs surrogate models, uncertainty, and acquisition functions.  
In Part 4, those ideas are implemented directly in a modern BO library.

We now study what happens when Bayesian Optimisation is treated not only as a conceptual framework, but as a **working computational workflow**:

- Gaussian Process surrogates are fit in BoTorch,
- acquisition functions are instantiated and optimised numerically,
- full sequential BO loops are built explicitly,
- and practical modelling choices are tested in settings where they visibly affect behaviour.

This part introduces the core implementation habits needed before more advanced BO settings can be understood:

- how BoTorch represents GP surrogates,
- how standard acquisition functions are evaluated and optimised,
- how a reusable BO loop is constructed,
- why minimisation problems are often recast in maximisation form,
- and how practical choices such as transforms, noise level, acquisition-optimisation settings, and initial design affect BO outcomes.

The goal is not just to learn *how to call BoTorch functions*, but to understand **how standard Bayesian Optimisation behaves once implemented in practice**.

---

## 🎯 Scope and Philosophy

Across these tutorials, BoTorch workflows are treated as:

- implementations of model-guided decision-making, not just API usage,
- numerical optimisation pipelines, not just mathematical abstractions,
- and practical tools whose behaviour depends strongly on modelling choices.

The material emphasises:

- the translation from Bayesian Optimisation concepts to BoTorch objects,
- the role of standard acquisition functions in decision-making,
- the construction of full sequential BO loops,
- the distinction between the true objective and the surrogate’s current belief,
- and the practical importance of scaling, noise, acquisition optimisation, and initial design.

By the end of Part 4, you should be comfortable answering questions like:

- *How do I implement and visualise standard acquisition functions in BoTorch?*
- *How does a full sequential BO loop look in code?*
- *Why can two BO runs with the same objective behave differently?*
- *What practical defaults usually make BoTorch BO behave better?*
- *Why does stronger acquisition optimisation not always imply better overall BO performance?*

---

## 📚 Contents

### Tutorial 1 — Gaussian Process Surrogates in BoTorch
**Focus:** Building, fitting, and querying GP surrogates in BoTorch.

- Constructing `SingleTaskGP`
- Input and outcome transforms
- Fitting hyperparameters with marginal likelihood
- Posterior mean and posterior variance in BoTorch
- Visualising surrogate fits in one dimension
- Interpreting GP behaviour through BoTorch objects

📓 `tutorial_01_gaussian_process_surrogates_in_botorch.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 2 — Standard Acquisition Functions in BoTorch
**Focus:** Turning a fitted surrogate into a next-point decision rule.

- Upper Confidence Bound (UCB)
- Probability of Improvement (PI)
- Log Expected Improvement (LogEI)
- Why BoTorch acquisition functions are commonly written for maximisation
- Recasting minimisation problems through `-f(x)`
- Visualising and optimising acquisition functions with `optimize_acqf`

📓 `tutorial_02_standard_acquisition_functions_in_botorch.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 3 — Building the Full Bayesian Optimisation Loop in BoTorch
**Focus:** Moving from one acquisition step to full sequential BO.

- Repeated surrogate fitting and acquisition optimisation
- Building a reusable single-step BO function
- Constructing the full sequential BO loop
- Best-observed-value trajectories over time
- Sampling-pattern evolution in one and two dimensions
- Interpreting BO as a path-dependent surrogate-driven process

📓 `tutorial_03_building_the_full_bayesian_optimisation_loop_in_botorch.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 4 — Practical Modelling Choices in BoTorch
**Focus:** Understanding which practical choices most strongly affect BO behaviour.

- Input normalisation and output standardisation
- Observation noise and its effect on surrogate quality
- Acquisition-optimisation settings such as `raw_samples` and `num_restarts`
- Initial design quality and sample efficiency
- Learned GP hyperparameters across BO iterations
- Practical defaults for stable and interpretable BoTorch BO

📓 `tutorial_04_practical_modelling_choices_in_botorch.ipynb`  
🛠 Worked version in `worked/`

---

## 🧠 How Part 4 Fits into the Series

Part 4 is **implementation-focused and workflow-oriented**.

It marks the point where the ideas of surrogate modelling, uncertainty, and acquisition-driven search are no longer treated only conceptually.  
Instead, they become explicit BoTorch workflows that can be run, inspected, and modified.

So the series now shifts:

- from the conceptual foundations of Bayesian Optimisation,
- to practical implementation and behaviour in a modern BO library.

This part intentionally stays within **standard single-loop Bayesian Optimisation** so that the behaviour of the workflow remains interpretable.

**Part 5** will extend these ideas further into more realistic and advanced settings, including:

- higher-dimensional experimental BO,
- batch BO for parallel experimentation,
- mixed-variable and constrained BO,
- and budget-aware or human-in-the-loop BO workflows.

---

## ✅ Prerequisites

- Completion of **Part 3**
- Basic familiarity with Gaussian Processes and acquisition functions
- Some comfort with PyTorch tensors and plotting
- Curiosity about *how Bayesian Optimisation ideas become real computational workflows in BoTorch*

---

**Author:** Angze Li  
**Version:** v1.0
