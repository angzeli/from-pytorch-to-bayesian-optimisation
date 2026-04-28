# 📘 Part 6 — Advanced Bayesian Optimisation Workflow Extensions

Part 6 takes the realism-oriented workflows of Part 5 and pushes them into settings where the optimiser must reason about **richer structure in the problem itself**.

In Part 5, the emphasis was on making BO operate in more realistic environments:  
higher-dimensional spaces, batch experimentation, mixed-variable and constrained design spaces, and workflows shaped by budget or human judgement.

In Part 6, those practical BO ideas are extended further into settings where the optimisation problem is no longer just a standard black-box loop over one flat design vector.

We now study what happens when Bayesian Optimisation must operate in workflows where:

- observations may be noisy and replication becomes part of the decision process,
- several objectives must be balanced at once through Pareto reasoning,
- evaluations may differ in fidelity and cost,
- recommendations may depend on an external context,
- and the experiment itself may have an internal hierarchy, with different variables playing different roles.

This part introduces the next layer of sophistication needed before BO can be treated as a realistic scientific decision-making framework rather than only an optimisation routine:

- how to reason about noise, uncertainty, and replication in the observed data,
- how to move from single-objective BO to multi-objective trade-off learning,
- how to decide not only **what** to evaluate, but also **how accurately** to evaluate it,
- how to make BO recommendations conditional on external operating conditions,
- and how to exploit internal workflow structure instead of treating every variable as interchangeable.

The goal is not just to make BO *more advanced*, but to understand **how BO behaviour changes once modelling assumptions, decision structure, and workflow hierarchy become part of the problem**.

---

## 🎯 Scope and Philosophy

Across these tutorials, Bayesian Optimisation is treated as:

- a scientific decision-making workflow, not just a next-point optimiser,
- a method whose behaviour depends strongly on how the objective is modelled and how information is collected,
- and a tool that must often respect the structure of the experimental process rather than only the geometry of the input space.

The material emphasises:

- the distinction between latent objective structure and observed noisy data,
- the fact that good BO decisions may involve replication as well as exploration,
- the difference between learning one optimum and learning a Pareto frontier,
- the fact that augmented inputs can mean very different things depending on whether they represent fidelity or context,
- and the importance of recognising when an experimental workflow is hierarchical rather than flat.

By the end of Part 6, you should be comfortable answering questions like:

- *How should BO be adapted when the data are noisy and repeated measurements matter?*
- *What changes once several objectives must be balanced instead of just one?*
- *How can BO use cheap low-fidelity evaluations without confusing them with the true target?*
- *Why is contextual BO not the same as multi-fidelity BO, even though both use augmented inputs?*
- *What does it mean for BO to respect a synthesis–operation hierarchy rather than treating all variables as one flat vector?*

---

## 📚 Contents

### Tutorial 1 — Noisy and Replication-Aware BO
**Focus:** Studying BO when observations are noisy and repeated evaluations become part of the optimisation logic.

- Defining noisy synthetic objectives
- Distinguishing latent objective value from noisy observation
- Tracking incumbent quality under noise
- Comparing BO against baselines in noisy settings
- Studying when replication is useful
- Building intuition for replication-aware decision-making

📓 `tutorial_01_noisy_and_replication_aware_bo.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 2 — Multi-Objective BO and Pareto-Optimal Decision-Making
**Focus:** Extending BO from single-objective optimisation to trade-off learning over several objectives.

- Defining synthetic multi-objective optimisation problems
- Identifying Pareto-optimal observations
- Visualising Pareto frontiers
- Using hypervolume as a summary of progress
- Applying qEHVI in BoTorch
- Distinguishing Pareto learning from final preference-based decision-making

📓 `tutorial_02_multi_objective_bo_and_pareto_optimal_dicision_making.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 3 — Multi-Fidelity and Contextual BO
**Focus:** Studying BO when evaluation accuracy varies by fidelity, or when the best design depends on an external context.

- Defining a synthetic high-fidelity target and lower-fidelity approximations
- Assigning evaluation cost as a function of fidelity
- Comparing high-fidelity-only BO against a multi-fidelity acquisition-per-cost workflow
- Distinguishing final high-fidelity recommendation quality from cheap proxy information
- Defining a contextual objective with a shifting optimum
- Running BO with context-dependent design recommendations

📓 `tutorial_03_multi_fidelity_and_contextual_bo.ipynb`  
🛠 Worked version in `worked/`

---

### Tutorial 4 — Structured BO for Hierarchical Experimental Workflows
**Focus:** Studying BO when the workflow itself has internal structure, so not all variables play the same role.

- Defining synthesis and operation variables separately
- Building a hierarchical synthetic objective
- Mapping synthesis conditions to latent material properties
- Defining preferred operation conditions conditional on synthesis
- Comparing flat BO against two-stage structured BO
- Analysing both performance discovery and synthesis-space exploration

📓 `tutorial_04_structured_bo_for_hierarchical_experimental_workflows.ipynb`  
🛠 Worked version in `worked/`

---

## 🧠 How Part 6 Fits into the Series

Part 6 is **structure-focused and modelling-aware**.

It marks the point where Bayesian Optimisation stops being only about adapting to more realistic optimisation environments and starts being treated as a framework whose behaviour depends on the deeper structure of the statistical and experimental problem.

So the series now shifts:

- from BO in more realistic practical settings,
- to BO in richer modelling and decision-making regimes.

This part intentionally keeps the same broad BO backbone:

- surrogate model,
- acquisition rule,
- proposal step,
- objective evaluation,
- and iterative update,

but repeatedly changes the **meaning of the observations, objectives, or inputs**.

That makes it possible to study how BO behaviour changes when:

- data are noisy and replication matters,
- the target is multi-objective rather than scalar,
- evaluations differ in fidelity and cost,
- recommendations must adapt to context,
- or the experiment itself is organised hierarchically.

---

## ✅ Prerequisites

- Completion of **Part 5**
- Familiarity with Gaussian Process surrogates and acquisition functions in BoTorch
- Comfort with the standard sequential BO loop
- Familiarity with PyTorch tensors, numerical optimisation, and plotting
- Some curiosity about how BO changes once noise, trade-offs, fidelity, context, and workflow structure all become part of the optimisation problem

---

**Author:** Angze Li  
**Version:** v1.0
