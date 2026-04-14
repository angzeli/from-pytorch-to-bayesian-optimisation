# 🧭 Roadmap: Part 6 and Beyond

> **This section outlines the roadmap for the rest of the repository.**  
> It gives a high-level view of what comes next, how the pieces connect, and where the project is ultimately heading.

This repository began with the foundations of tensors, gradients, and optimisation, and then moved through Gaussian Processes and Bayesian Optimisation.  
The next stage is to take those ideas beyond standard toy workflows and into settings that look much more like real scientific and experimental decision problems.

---

# 📘 Part 6 — Advanced Bayesian Optimisation Extensions

> **Goal: extend Bayesian Optimisation into richer, more realistic, and more research-oriented settings.**

If Part 5 focuses on bringing BO closer to realistic experimental workflows, Part 6 pushes further into important extensions that arise when optimisation becomes noisier, more structured, and more complex.

These tutorials move beyond standard single-objective, single-fidelity, continuous-space BO and introduce settings that are especially relevant in scientific and experimental optimisation.

## Structure of Part 6

### Tutorial 1 — Noisy and replication-aware BO
Extend BO to settings where observations are noisy and repeated measurements may be valuable.

This tutorial focuses on:

- noise-aware acquisition strategies,
- the effect of observation uncertainty on BO behaviour,
- and the trade-off between exploring new points and replicating previous ones to reduce uncertainty.

---

### Tutorial 2 — Multi-objective BO
Move from optimising a single target to balancing multiple competing objectives.

Examples include simultaneously optimising:

- performance,
- stability,
- selectivity,
- cost,
- or other competing experimental criteria.

This tutorial introduces the idea of Pareto-optimal decision-making in BO.

---

### Tutorial 3 — Multi-fidelity and contextual BO
Extend BO to settings where evaluations can occur at different fidelities, or where optimisation depends on external context.

This tutorial focuses on:

- trading off cheap approximate evaluations against expensive accurate ones,
- incorporating contextual information into the BO workflow,
- and making decisions when the objective depends not only on chosen variables but also on the surrounding setting.

---

### Tutorial 4 — Combinatorial and structured BO
Study BO in design spaces that are no longer simple continuous vectors.

Examples include optimisation over:

- discrete combinations,
- categorical structures,
- sequences,
- graphs,
- or other structured experimental choices.

This tutorial focuses on how BO changes when the search space itself becomes combinatorial or structured.

---

# 🚀 Beyond BO

After Bayesian Optimisation, there are several natural directions this project can continue toward.

These directions may develop **outside this repository**, as separate but related projects, depending on whether the focus shifts more toward machine learning workflows or scientific computing and visual communication.

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
