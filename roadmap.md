# 🧭 Roadmap: Part 5 and Beyond

> **This section outlines the roadmap for the rest of the repository.**  
> It gives a high-level view of what comes next, how the pieces connect, and where the project is ultimately heading.

This repository began with the foundations of tensors, gradients, and optimisation, and then moved through Gaussian Processes and Bayesian Optimisation.  
The next stage is to take those ideas beyond standard toy workflows and into settings that look much more like real scientific and experimental decision problems.

---

# 📘 Part 5 — Advanced Bayesian Optimisation

> **Goal: extend Bayesian Optimisation to more realistic, higher-value, and more experimentally relevant settings.**

So far, the repository has focused on the core BO workflow:

- fit a surrogate,
- construct an acquisition function,
- choose the next point,
- evaluate the objective,
- and repeat.

Part 5 asks what happens when this workflow is pushed closer to real applications, where optimisation is rarely:

- one-dimensional,
- perfectly sequential,
- purely continuous,
- or free of practical constraints.

Instead, real BO problems often involve:

- many decision variables,
- parallel experimentation,
- mixed continuous / discrete design spaces,
- limited evaluation budgets,
- and human judgement in the loop.

That is the focus of Part 5.

## Structure of Part 5

### Tutorial 1 — Higher-dimensional custom BO for experimental design spaces
Move from low-dimensional toy examples to genuinely higher-dimensional BO problems motivated by experimental optimisation.

Examples include situations where performance depends on many controllable factors, such as:

- synthesis conditions,
- operating conditions,
- composition variables,
- processing parameters,
- and other experimental degrees of freedom.

This tutorial focuses on how to formulate and run BO in richer continuous design spaces.

---

### Tutorial 2 — Batch BO for parallel experimentation
Extend BO from choosing **one point at a time** to choosing **multiple points in a single iteration**.

This is especially relevant when experiments or simulations can be run in parallel, for example:

- multiple reactions in one batch,
- several simulations on a compute cluster,
- or several candidate conditions tested in the same cycle.

The emphasis here is on practical batch selection and the trade-off between sequential and parallel decision-making.

---

### Tutorial 3 — Mixed-variable and constrained BO
Introduce BO in more realistic design spaces containing:

- continuous variables,
- integer variables,
- categorical choices,
- and feasibility constraints.

This reflects the fact that many real optimisation problems do not live in simple unconstrained continuous domains.

The tutorial focuses on how these complications change the modelling and decision process.

---

### Tutorial 4 — Budget-aware and human-in-the-loop BO workflows
Study BO in settings where optimisation must respect real practical limits, such as:

- a fixed experiment budget,
- limited material or time,
- or the need for expert filtering and scientific judgement.

This tutorial emphasises that BO is often most useful not as a fully autonomous system, but as a **decision-support tool** used alongside human expertise.

---

# 🚀 Beyond BO

After Bayesian Optimisation, the repository can branch in more than one direction.

Both directions are natural continuations, depending on whether the emphasis shifts toward **machine learning systems** or **scientific computing and communication**.

## Option A — Applied Machine Learning with PyTorch
This path moves from optimisation and modelling foundations toward broader machine learning workflows.

Possible topics include:

- datasets and data pipelines,
- training loops and validation,
- practical model development,
- and architecture-level reasoning.

This direction is a natural continuation if the project grows toward predictive modelling, representation learning, or applied machine learning more broadly.

---

## Option B — Scientific Visualisation and Data Analysis with matplotlib
This path develops the communication and analysis side of scientific computing.

Possible topics include:

- publication-quality figures,
- advanced plotting and layout techniques,
- diagnostic visualisation,
- and reproducible data-analysis workflows.

This direction is especially natural if the project continues to emphasise scientific interpretation, experiment analysis, and research presentation.

---

# 🧠 Big Picture

The repository follows a coherent progression:

- **Part 1 — Foundations** → how we represent and manipulate mathematical objects  
- **Part 2 — Optimisation** → how systems move under gradient-based updates  
- **Part 3 — Modelling** → how we represent unknown functions and uncertainty  
- **Part 4 — Bayesian Optimisation** → how modelling and optimisation combine into decision-making  
- **Part 5 — Advanced Bayesian Optimisation** → how that decision-making framework is extended to realistic and practical settings  

So the overall direction of the repository is not just to learn PyTorch or Bayesian Optimisation in isolation.

It is to build a layered understanding of:

- representation,
- optimisation,
- uncertainty,
- and decision-making

as connected parts of a broader scientific and computational workflow.

---

## Final Perspective

> **Optimisation alone is not enough.**  
> **Modelling alone is not enough.**

What makes Bayesian Optimisation powerful is their combination:

> **using models of uncertainty to make efficient decisions under limited budgets.**

That is the central idea connecting the whole repository so far — and Part 5 is where that idea begins to look much more like the real problems it was designed for.
