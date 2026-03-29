# 📘 Gradients, Autograd, and Optimisation in PyTorch

This repository is a concept-first exploration of gradients, automatic differentiation, and optimisation using PyTorch.

Rather than teaching PyTorch as a high-level deep-learning framework, the material treats it as a numerical and analytical tool for understanding:
- tensors and linear algebra,
- computation graphs and automatic differentiation,
- gradient structure and sensitivity,
- and how optimisation emerges from repeated gradient-based updates.

The goal is not just to show how to compute gradients, but to understand **what they mean, where they come from, and how they shape learning dynamics**.

---

## 🧠 Philosophy

Most tutorials jump quickly from tensors to neural networks to training loops.

This repository deliberately slows down.

Gradients are treated as:
- mathematical objects, not black-box training signals,
- sensitivity measures, not just optimisation updates,
- structured quantities shaped by objectives and model architecture.

By the time optimisers are introduced, they should feel like a natural consequence of ideas already developed — not a new abstraction to memorise.

---

## 📦 Repository Structure

The material is organised into Parts, each forming a coherent conceptual unit.
```python
├── part_1/
│   ├── worked/
│       └── (worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_tensor_fundamentals.ipynb
│   ├── tutorial_02_minimal_learning_problem.ipynb
│   ├── tutorial_03_autograd_and_graphs.ipynb
│   ├── tutorial_04_tensor_gradients_and_vjp.ipynb
│   ├── workshop_01_gradient_flow_to_optimisation.ipynb
│   └── workshop_02_objective_design_and_local_dynamics.ipynb
├── part_2/ #currently working
│   ├── worked/
│       └── (worked and exploratory versions)
│   ├── README.md
│   ├── tutorial_01_gradient_as_dynamical_system.ipynb
│   ├── tutorial_02_geometry_and_conditioning_of_optimisation.ipynb
│   ├── tutorial_03_momentum_as_a_dynamical_system.ipynb
│   ├── tutorial_04_optimisation_beyond_convexity.ipynb
│   ├── workshop_01 #currently working
│   └── workshop_02 #currently working
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

## 🚧 Part 2 — Optimisation Dynamics (Coming Next)

Part 2 will build directly on the gradient intuition developed in Part 1 and introduce:
- gradient descent as repeated local updates,
- learning rates, conditioning, and scaling,
- curvature, geometry, and stability,
- and how optimisation behaviour emerges over time.

Nothing in Part 2 will assume unfamiliar gradient concepts — the groundwork is laid entirely in Part 1.

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

## 🧭 How to Use This Repository

Recommended order:
	1.	Work through Tutorials 1–4 sequentially.
	2.	Use Workshop 1 to consolidate gradient intuition.
	3.	Use Workshop 2 to bridge gradients to optimisation thinking.
	4.	Continue into Part 2 when available.

You are encouraged to:
- modify cells,
- change upstream gradients,
- visualise `.grad`,
- and treat the notebooks as experimental sandboxes.

---

## ✍️ Author & Status

**Author**: Angze Li

**Status**: Actively developed

**Last updated**: 2026-02-20
