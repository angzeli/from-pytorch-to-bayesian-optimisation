# 📘 From PyTorch to Bayesian Optimisation

This repository is a concept-first exploration of gradients, automatic differentiation, optimisation, and surrogate-based decision-making using PyTorch.

Rather than teaching PyTorch only as a high-level deep-learning framework, the material uses it as a numerical and analytical tool for understanding a much broader progression:

- tensors and linear algebra,
- computation graphs and automatic differentiation,
- gradient structure and optimisation dynamics,
- modelling unknown objective functions,
- uncertainty-aware surrogate models,
- and finally **Bayesian Optimisation** through BoTorch.

The goal is not just to show how to compute gradients or train models, but to build a coherent path from **low-level PyTorch mechanics** to **modern data-efficient optimisation**.

In other words, this repository is designed as a bridge:

> **from PyTorch fundamentals to Bayesian Optimisation.**

---

## 🧠 Philosophy

Most tutorials either:
- stop at PyTorch basics,
- jump quickly into neural-network training,
- or treat Bayesian Optimisation as a separate black-box topic.

This repository takes a different route.

It deliberately slows down and builds the ideas in sequence.

PyTorch is treated not just as a framework for fitting models, but as a flexible environment for understanding:
- gradients as mathematical objects,
- autograd as a computational mechanism,
- optimisation as a dynamical process,
- and surrogate modelling as a way to reason about expensive unknown functions.

By the time Bayesian Optimisation is introduced, it should feel like the natural outcome of ideas already developed:
- first understand gradients,
- then understand optimisation,
- then understand why optimisation alone is not enough,
- then build models of unknown functions,
- and finally use those models to guide intelligent search.

The aim is therefore not just to teach isolated tools, but to build a conceptual pathway:

> **PyTorch → gradients → optimisation → surrogate modelling → Bayesian Optimisation**

---

## 📦 Repository Structure

The material is organised into Parts, each forming a coherent conceptual unit.
```python
├── part_1/
│   ├── worked/ #(worked and exploratory versions)
│   ├── README.md 
│   ├── tutorial_01_tensor_fundamentals.ipynb
│   ├── tutorial_02_common_pytorch_tensor_operations.ipynb
│   ├── tutorial_03_minimal_learning_problem.ipynb
│   ├── tutorial_04_autograd_and_graphs.ipynb
│   ├── tutorial_05_tensor_gradients_and_vjp.ipynb
├── part_2/
│   ├── worked/ #(worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_gradient_as_dynamical_system.ipynb
│   ├── tutorial_02_geometry_and_conditioning_of_optimisation.ipynb
│   ├── tutorial_03_momentum_as_a_dynamical_system.ipynb
│   ├── tutorial_04_optimisation_beyond_convexity.ipynb
├── part_3/
│   ├── worked/ #(worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_why_model_an_unknown_function.ipynb
│   ├── tutorial_02_prediction_uncertainty_and_confidence.ipynb
│   ├── tutorial_03_gaussian_processes_as_surrogate_models.ipynb
│   └── tutorial_04_choosing_where_to_evaluate_next.ipynb
├── part_4/ 
│   ├── worked/ #(worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_from_gaussian_processes_to_botorch_model.ipynb
│   ├── tutorial_02_standard_acquisition_functions_in_botorch.ipynb
│   ├── tutorial_03_full_single_loop_bo_workflow.ipynb
│   └── tutorial_04_practical_modelling_choices_in_botorch.ipynb
├── part_5/ 
│   ├── worked/ #(worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_higher_dimensional_custom_bo_for_experimental_design_spaces.ipynb
│   ├── tutorial_02_batch_bo_for_parallel_experimentation.ipynb
│   ├── tutorial_03_mixed_variable_and_constrained_bo.ipynb
│   └── tutorial_04_budget_aware_and_human_in_the_loop_bo_workflows.ipynb
├── part_6/ 
│   ├── worked/ #(worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_noisy_and_replication_aware_bo.ipynb
│   ├── tutorial_02_multi_objective_bo.ipynb #currently working
│   ├── tutorial_03_multi_fidelity_and_contextual_bo.ipynb #currently working
│   └── tutorial_04_combinatorial_and_structured_bo.ipynb #currently working
├── LICENSE
└── README.md
```
Each notebook is self-contained and can be read independently, but the intended experience is sequential.

🌱 The repository is still growing.

---
## 📘 Part 1 — Foundations: Tensors, Autograd, and Gradient Structure

Part 1 builds the conceptual foundations needed to understand gradients before optimisation algorithms are introduced.

It covers:
- tensor mechanics and numerical structure,
- how autograd builds and traverses computation graphs,
- scalar vs tensor-valued differentiation,
- vector–Jacobian products as the core object of `backward`,
- interpreting `.grad` as sensitivity,
- and visualising gradient structure in controlled experiments.

Part 1 concludes with two workshops that explicitly connect gradient structure to **local optimisation intuition**, without yet introducing optimisers or training pipelines.

📂 See `part_1/README.md` for full details.

---

## 📘 Part 2 — Optimisation Dynamics 

Part 2 builds directly on the gradient intuition developed in Part 1 and studies how optimisation behaviour emerges over time.

It covers:
- gradient descent as a discrete dynamical system,
- learning rates, stability, and contraction,
- geometry, conditioning, and narrow valleys,
- momentum and inertia,
- and the challenges of optimisation beyond convexity.


📂 See `part_2/README.md` for full details.

---

## 📘 Part 3 — Modelling Unknown Functions and Bayesian Optimisation Foundations

Part 3 is the conceptual bridge from optimisation dynamics to **Bayesian Optimisation**.

We now study what happens when the objective function is:
- expensive to evaluate,
- only partially observed,
- and better handled through a learned surrogate than through brute-force search.

This part develops the core ideas needed before using modern Bayesian Optimisation libraries.

It introduces:
- why expensive objectives require modelling,
- how surrogate models approximate unknown functions,
- why prediction alone is not enough without uncertainty,
- Gaussian Processes as principled probabilistic surrogates,
- and acquisition functions for deciding where to evaluate next.

This prepares the ground for the next stage of the repository, where these ideas are implemented more practically using **BoTorch**.

📂 See `part_3/README.md` for full details.

---

## 📘 Part 4 — Practical Bayesian Optimisation with BoTorch

Part 4 turns the conceptual foundations of Part 3 into practical workflows using **BoTorch**.

Rather than building Gaussian Processes and acquisition logic entirely from first principles, we now study how those same ideas are implemented in a modern Bayesian Optimisation library.

It covers:
- fitting Gaussian Process surrogates in BoTorch,
- working with BoTorch posterior objects,
- acquisition functions such as EI, PI, and UCB,
- optimising acquisition functions to propose new candidates,
- and building the standard sequential Bayesian Optimisation loop in practice.

Part 4 is still focused on **standard single-loop Bayesian Optimisation**.
Its purpose is to make the transition from theory to implementation clear and interpretable, before moving on to more advanced BO strategies.

📂 See `part_4/README.md` for full details.

---

## 📘 Part 5 — Advanced Bayesian Optimisation Workflows

Part 5 extends the standard BoTorch workflows from Part 4 into more realistic experimental optimisation settings.

It covers:
- higher-dimensional BO for experimental design spaces,
- batch BO for parallel experimentation,
- mixed-variable and constrained BO,
- decode–repair–evaluate workflows,
- budget-aware BO under unequal experiment costs,
- and human-in-the-loop BO with simple decision rules.

Part 5 focuses on the idea that practical BO is not only about maximising an acquisition function. In realistic workflows, candidate selection may also depend on feasibility, cost, budget, and human judgement.

This part therefore bridges standard single-loop BO and more realistic scientific optimisation campaigns.

📂 See `part_5/README.md` for full details.

---

## 🧪 Fresh vs Worked Notebooks

For most notebooks, two versions exist:
- "Fresh": clean, minimal versions intended for reading, teaching, or first-pass study.
  The "Fresh" notebooks are in the main folder.
  
- Worked: exploratory versions containing additional experiments, plots, and intermediate outputs. Models answers are also in these notebooks.
  The worked notebooks are in `worked/` folder.

This separation keeps the main narrative clear while preserving the full reasoning process.

---

## 🎯 Intended Audience

This repository is suitable for:
- advanced undergraduates,
- master’s students,
- PhD students,
- or practitioners who want a deeper understanding of gradients and optimisation.

A background in linear algebra and basic calculus is assumed, but no prior deep-learning experience is required.

---

## ✍️ Author & Status

**Author**: Angze Li

**Status**: Actively developed
