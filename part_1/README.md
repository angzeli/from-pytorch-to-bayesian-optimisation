# 📘 Part 1 — PyTorch Foundations: Tensors, Autograd, and Gradient Structure

Part 1 introduces the conceptual foundations needed to understand **how tensors are manipulated, how gradients are constructed, how they propagate through computation graphs, and how they should be interpreted in PyTorch** before any optimisation algorithms are introduced.

Rather than focusing on training pipelines or neural network abstractions, this part treats PyTorch as a **numerical and analytical framework** for studying tensor structure, automatic differentiation, and gradient flow.

The aim is not only to learn *how* to compute gradients, but to understand **what they mean, where they come from, and why they take the form they do**.

---

## 🎯 Scope and Philosophy

Across these tutorials, gradients are treated as:

- mathematical objects, not black-box signals,
- sensitivity measures, not just optimisation updates,
- structured quantities shaped by both computation graphs and chosen objectives.

The material emphasises:

- explicit reasoning about tensor shape and tensor operations,
- computation graphs as the mechanism underlying `autograd`,
- scalar and tensor-valued differentiation,
- vector–Jacobian products as the core object computed by `backward(v)`,
- and interpreting `.grad` geometrically, structurally, and analytically.

By the end of Part 1, you should be comfortable answering questions such as:

- *What exactly does `.backward()` compute?*
- *Why are tensor shape and dimensionality so important in PyTorch?*
- *How do computation graphs determine gradient flow?*
- *Why does a gradient look the way it does?*
- *What changes when the output is not a scalar?*

---

## 📚 Contents

### Tutorial 1 — Tensor Fundamentals
**Focus:** Core tensor mechanics and numerical intuition.

- Tensor creation from Python and NumPy
- Shapes, dimensions, and indexing semantics
- Tensor attributes such as `shape`, `dtype`, and `device`
- Element-wise vs matrix operations
- Aggregation, scalar tensors, and `.item()`
- In-place operations and common pitfalls

📓 `tutorial_01_tensor_fundamentals.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 2 — Common PyTorch Tensor Operations
**Focus:** Practical tensor manipulation and essential utility operations.

- Reshaping with `squeeze()` and `unsqueeze()`
- Constructing structured tensors with `torch.linspace()` and `torch.eye()`
- Concatenation with `torch.cat()`
- Querying extrema with `torch.argmin()` and `torch.argmax()`
- Conditional tensor logic with `torch.where()`
- Statistical reduction with `torch.std()`

📓 `tutorial_02_common_pytorch_tensor_operations.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 3 — A Minimal Learning Problem
**Focus:** First encounter with gradients and learning dynamics.

- Defining a minimal parametric model
- Forward pass and scalar loss construction
- Trainable parameters and `requires_grad=True`
- What `loss.backward()` actually computes
- Manual gradient updates and learning-rate effects
- The basic structure of gradient-based learning

📓 `tutorial_03_minimal_learning_problem.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 4 — Autograd and Computation Graphs
**Focus:** How gradients are produced from computation graphs.

- Directed acyclic graphs (DAGs) in `autograd`
- `grad_fn`, `next_functions`, and graph traversal
- Leaf vs intermediate tensors
- Gradient accumulation and graph reuse
- Higher-order differentiation with `create_graph=True`
- Understanding backpropagation as structured reverse computation

📓 `tutorial_04_autograd_and_graphs.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 5 — Tensor Gradients and Vector–Jacobian Products
**Focus:** Differentiation beyond scalar losses.

- Scalar vs tensor-valued outputs
- Why full Jacobians are rarely constructed explicitly
- Vector–Jacobian products via `backward(v)`
- Explicit upstream gradients as sensitivity weights
- Interpreting `.grad` for non-scalar outputs
- Visualising how gradient structure changes with output weighting

📓 `tutorial_05_tensor_gradients_and_vjp.ipynb`  
🛠 Worked version in `worked/`

---

## 🧠 How Part 1 Fits into the Series

Part 1 is **conceptual, structural, and diagnostic**.

It deliberately avoids:

- optimisers,
- full training pipelines,
- datasets,
- and neural network modules as the main focus.

Instead, it builds a mental model of PyTorch in layers:

1. tensors as structured mathematical objects,
2. tensor operations as the practical language of computation,
3. gradients as derivatives of tracked computations,
4. computation graphs as the mechanism behind backpropagation,
5. and vector–Jacobian products as the correct framework for non-scalar differentiation.

This is what makes later topics — such as gradient descent, optimisation dynamics, curvature, and objective design — feel **motivated rather than mysterious**.

**Part 2** will build on this by taking gradients as already-understood objects and asking what happens when we use them repeatedly to drive optimisation.

---

## ✅ Prerequisites

- Basic Python
- Elementary linear algebra
- Some familiarity with vectors, matrices, and functions
- Curiosity about *why* gradients behave the way they do

---

**Author:** Angze Li
**Version:** v2.0
