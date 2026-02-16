# Technical Audit: The "GlassBox" Symbolic Regression Architecture

## Critical Analysis of Feature Engineering, Generative Priors, and Search Strategies in the Era of Neuro-Symbolic Foundation Models

### Executive Summary

The "GlassBox" (or "Principia") project represents a sophisticated implementation of classical Neuro-Symbolic architecture, hybridizing a feature-based classification system with an Operational Neural Network (ONN) to solve the Symbolic Regression (SR) problem. The pipeline, as defined in the provided algorithmic documentation, relies on a distinct two-stage methodology: extracting a 334-dimensional vector of handcrafted statistical and spectral features to classify the operator composition of a curve, followed by a warm-started evolutionary search and a fast-path regression solver. This approach mirrors the "feature-engineering first" paradigm that dominated machine learning applications in the physical sciences between 2015 and 2020. It effectively operationalizes domain intuition—specifically the notion that mathematical functions possess distinct "fingerprints" in their shape, frequency content, and derivatives—to prune the combinatorial search space of genetic programming.

However, a rigorously critical review of this architecture against the research landscape of 2024–2025 reveals fundamental structural obsolescence. The field of Symbolic Regression has undergone a phase transition, shifting from heuristic feature engineering to **End-to-End (E2E) Representation Learning**, and from soft probabilistic biasing to **Structural Seeding** and **Foundation Model Distillation**. The reliance on 334 handcrafted scalars introduces a quantifiable "Information Bottleneck," discarding phase coherence, topological invariance, and high-frequency structural details that are essential for recovering complex, nested analytical expressions. Furthermore, the "1D Slicing" strategy for multivariate functions constitutes a mathematically fragile heuristic that fails catastrophically for non-separable interactions—a class of functions that comprises the majority of real-world physical laws.

This report provides an exhaustive, ruthless critique of the GlassBox pipeline. It contrasts the current bottlenecks with State-of-the-Art (SOTA) methodologies, including **Multimodal Symbolic Regression (MMSR)** 1, **Structure-Aware Symbolic Regression (StruSR)** 2, and **Partially Initialized Genetic Programming (PIGP)**.3 The analysis is structured to dismantle the existing assumptions regarding feature sufficiency, generative bias, and multivariate decomposition, offering a concrete modernization roadmap to align GlassBox with the frontier of Neuro-Symbolic AI.

## 1\. Critique of Feature Engineering vs. Representation Learning

The core premise of the current Curve Classifier is that the functional identity of a dataset $(X, y)$ can be compressed into a fixed-length vector of 334 scalars.4 This vector includes 128 raw shape points, 32 FFT magnitudes, 128 derivative points, and 46 statistical/curvature metrics. While this approach captures low-frequency behavioral trends, it suffers from severe information loss when compared to modern deep learning architectures that ingest raw data directly.

### 1.1 The Information Bottleneck in Handcrafted Features

The theory of **Sufficient Statistics** in statistical inference asks whether a summary statistic $T(X)$ preserves all information in $X$ about a parameter $\theta$. In Symbolic Regression, $\theta$ is the symbolic expression itself. The 334 features employed in GlassBox are demonstrably _insufficient statistics_ for the space of analytical functions, primarily due to loss of phase, derivative noise amplification, and lack of permutation invariance.

#### 1.1.1 Spectral Decomposition and Phase Erasure

The reliance on **FFT Magnitudes (32 dims)** 4 is perhaps the most critical loss point. The magnitude spectrum $| \mathcal{F}(f) |$ captures the energy distribution across frequencies but discards the **Phase Spectrum** $\angle \mathcal{F}(f)$. In physical laws and mathematical identities, phase carries structural information. Consider the distinction between a sum of sines and a modulated sine wave:

$$f_1(x) = \sin(x) + \sin(3x)$$

$$f_2(x) = \sin(x) \cdot \sin(3x) = 0.5(\cos(2x) - \cos(4x))$$

While these functions have different spectral fingerprints, the phase relationships in composite functions (like square waves vs. triangle waves constructed from the same harmonics) determine the sharp transitions and extrema locations. By discarding phase, the classifier loses the ability to distinguish between functions that share energetic properties but differ topologically. Modern Transformer architectures, such as **DeepSym** or **NeSymRes**, process the sequence of points directly. They learn to attend to local structures (like a sharp peak or a discontinuity) that are "smeared out" in the frequency domain. The attention mechanism $Attention(Q, K, V)$ effectively learns a **Time-Frequency representation** (similar to a wavelet transform but learned) that preserves both local and global structure, a capability mathematically impossible for a fixed FFT magnitude vector.

#### 1.1.2 The Instability of Numerical Derivatives

The pipeline uses 128 points of 1st and 2nd derivatives. Numerical differentiation is an **ill-posed inverse problem** in the presence of noise. Even with sophisticated smoothing (e.g., Savitzky-Golay filters), the derivative features introduce a hyperparameter trade-off: aggressive smoothing obliterates high-frequency symbolic terms (like $\sin(100x)$ inside a generic envelope), while weak smoothing allows noise to dominate the feature vector. Recent work in **StruSR (Structure-Aware Symbolic Regression)** 2 demonstrates that direct numerical derivatives are suboptimal. StruSR employs **Physics-Informed Neural Networks (PINNs)** to learn a continuous surrogate model of the data first, from which derivatives are extracted analytically via automatic differentiation. This "denoising-through-modeling" step ensures that the derivative features fed into the symbolic search are mathematically consistent with the underlying manifold, rather than artifacts of finite-difference errors. The GlassBox approach of computing derivatives directly on raw (or simply smoothed) data likely injects high variance into the classifier, confusing the operator probabilities for terms like exp or log which are sensitive to derivative behavior at boundaries.

#### 1.1.3 Invariance and the Curse of Dimensionality

The feature vector is fixed at 334 dimensions. This fixed allocation is problematic for multi-scale functions. A function might have critical behavior concentrated in a tiny region of the domain (e.g., a resonance peak or a singularity). In a 128-point resampled "Shape" vector, this feature might occupy only 1 or 2 bins, effectively vanishing against the background signal. Contrast this with **PointNet** or **Set Transformer** architectures.5 These models treat the input as a set of $\{(x_i, y_i)\}$ tuples. They apply a symmetric function (like max-pooling) over a high-dimensional learned embedding space (e.g., 1024 dimensions per point). This allows the network to "focus" its entire representational capacity on a single outlier point if that point carries critical information (like a vertical asymptote indicating a rational function with a root in the denominator). The handcrafted feature vector essentially "averages out" these critical local singularities, rendering the classifier blind to the very features that define rational and hyperbolic function classes.

### 1.2 The Paradigm Shift: End-to-End Representation Learning

The research literature from 2024 and 2025 has decisively moved away from feature engineering toward **End-to-End (E2E) Learning**. This shift is driven by the realization that neural networks can learn a "Language of Functions" that is far richer than any human-designed feature set.

#### 1.2.1 Transformer Encoders as Universal Approximators

The **E2E Symbolic Regression** framework (Kamienny et al., 2022; updated 2025) 7 utilizes a Transformer Encoder to map the sequence of input points to a latent embedding $z$. This embedding is not a statistical summary; it is a **semantically rich vector** in a continuous space where "functions that look alike" are clustered together.

*   **Mechanism:** The Transformer uses self-attention to relate every point $(x_i, y_i)$ to every other point $(x_j, y_j)$. This $N^2$ interaction map allows the model to detect global symmetries (e.g., $f(x) = f(-x)$) and repeating patterns (periodicity) without explicit programming.
*   **Performance:** Empirical results from **SRBench 2025** 9 show that E2E Transformers achieve higher recovery rates on Feynman and Strogatz datasets compared to feature-based classifiers, specifically because they are robust to variable input densities and distributions. They do not require resampling to a fixed 128-point grid, allowing them to handle irregular or sparse data natively.

#### 1.2.2 Multimodal Contrastive Learning (MMSR)

The most advanced critique of the GlassBox feature extractor comes from the **MMSR (Multimodal Symbolic Regression)** architecture.1 MMSR posits that Symbolic Regression is a **Multimodal Alignment Task** between the "Numeric Modality" (the data points) and the "Symbolic Modality" (the equation string).

*   **The Architecture:** MMSR trains two encoders: a Data Encoder (for $X, y$) and a Skeleton Encoder (for the equation tree).
*   **The Objective:** It uses **Contrastive Loss (InfoNCE)** to align the latent representations. The embedding of the data for $y = \sin(x)$ is forced to be identical to the embedding of the symbolic token sequence [sin, x].
*   **Implication for GlassBox:** Your current classifier maps features to _classes_ (probabilities of operators). MMSR maps data to a _semantics_ (a location in equation space). This enables **Zero-Shot Generalization**: if the model encounters a function it hasn't seen before, it projects it to the nearest valid equation embedding, rather than failing to classify it into a pre-defined bucket. The handcrafted features of GlassBox create a rigid "classification boundary" that lacks this continuous semantic interpolation.

### 1.3 Theoretical Superiority of PointNet Encoders

You specifically queried the comparison with **PointNet**. In the context of symbolic regression, PointNet architectures offer a theoretical guarantee of **Permutation Invariance** that handcrafted features (which usually assume sorted $x$) do not fully utilize.

*   **Set Theory:** A dataset is a set, not a sequence. Imposing an order (sorting by $x$) is a heuristic that aids FFT but fails in higher dimensions ($N > 1$) where no natural sorting exists.
*   **PointNet Mechanism:** A PointNet computes $h(x_i, y_i)$ for each point and then aggregates via $g(\{h_1, \dots, h_N\}) = \max(h_i)$. This architecture is fundamentally dimension-agnostic. It works identically for 1D, 2D, or 10D data, simply by changing the input dimension of the MLP.
*   **GlassBox Comparison:** The GlassBox reliance on "1D Slicing" (to be discussed in Section 3) is a direct consequence of using features (FFT/Derivatives) that fundamentally require 1D ordered data. Adopting a PointNet or Set Transformer encoder would allow GlassBox to ingest $N$-dimensional data natively, removing the need for the flawed slicing heuristic entirely.

## 2\. Analysis of Data Generation Strategy

The GlassBox pipeline uses specific template families (Simple, Compound, Rational, Physics-inspired) and AST parsing to generate training data.4 While this ensures high precision on known formula types, it introduces a severe **Inductive Bias** that limits generalization to novel scientific discoveries.

### 2.1 The Bias-Variance Trade-off in Template-Based Generation

Template-based generation creates a dataset that is a collection of "islands" in the vast ocean of mathematical expressions.

*   **The Bias:** By defining a "Rational" template as, say, $P_3(x) / Q_2(x)$, the model learns to recognize exactly this structure. If the real-world data follows a slightly different rational form—say, $\frac{e^x}{1+x^2}$—the classifier may fail to trigger the "Rational" class because the numerator is exponential, not polynomial.
*   **The "Memorization" Effect:** Research by **Voigt et al. (2025)** on "Analyzing Generalization in Pre-Trained Symbolic Regression" 8 explicitly studied this phenomenon. They found that models trained on restricted generative templates (like the Feynman dataset distribution) perform excellently In-Distribution (ID) but degrade catastrophically on Out-Of-Distribution (OOD) tasks (like the Black-Box Strogatz collection). The model essentially acts as a "nearest neighbor" look-up, mapping inputs to the closest template it memorized, rather than learning the compositional rules of mathematics.

### 2.2 Probabilistic Context-Free Grammars (PCFG)

Modern baselines (e.g., DeepSym, MMSR) utilize **Probabilistic Context-Free Grammars (PCFG)** or **Uniform Tree Sampling** to generate data.

*   **Grammar vs. Template:** A PCFG defines recursive rules (e.g., $E \to E + T$, $T \to \sin(F)$). This allows for infinite recursion and compositionality. A PCFG can generate $\sin(\cos(\sin(x)))$ even if the engineer never explicitly coded a "Nested Sine" template.
*   **Coverage:** PCFG-based datasets cover the "connective tissue" between the template islands. They expose the model to "weird" functions—unstable polynomials, deeply nested compositions, fractional exponents—that force the feature extractor to learn robust, generalized representations rather than template matching.
*   **Uniform Sampling:** To prevent the model from biasing toward short expressions, advanced generators use **uniform generation algorithms** (like those proposed by Lample & Charton) which ensure that trees of depth 5 are as likely as trees of depth 10, preventing the "simplicity bias" from causing underfitting on complex real-world data.

### 2.3 Foundation Model Distillation (EQUATE)

The cutting edge of 2025 data generation is represented by **EQUATE (Equation Generation via Quality-Aligned Transfer Embeddings)**.13 This method moves beyond both templates and PCFGs by using **LLM Distillation**.

*   **Concept:** Instead of writing a grammar, researchers query a Large Language Model (e.g., Llama-3 or GPT-4) to "generate 10,000 equations that might appear in a fluid dynamics textbook."
*   **Scientific Prior:** The LLM, having been trained on the entire corpus of arXiv and GitHub, possesses an implicit "Scientific Prior." It generates equations that respect dimensional homogeneity, use common variable names, and exhibit structures typical of physics (e.g., exponential decays, inverse square laws).
*   **Distillation:** These LLM-generated equations are then verified numerically and used to train the SR model.
*   **Superiority:** This approach creates a dataset that is **semantically aligned** with human scientific discovery. It avoids the "junk" equations often produced by PCFGs (like $\sin(\sin(\sin(\sin(x))))$) which are mathematically valid but physically unlikely. For GlassBox, replacing static templates with an **LLM-driven generative pipeline** would significantly improve the model's ability to "warm-start" on realistic scientific data.

### 2.4 Structural Bias and OOD Generalization

The user specifically asks about the limit on **Out-of-Distribution (OOD)** generalization.

*   **The Problem:** Neural networks interpolate well but extrapolate poorly. If your templates only contain polynomials of degree $\le 5$, the network effectively learns to approximate any function as a degree-5 polynomial (Taylor approximation bias). When faced with a degree-10 polynomial or a Bessel function, the network will confidently predict a degree-5 approximation, leading to high symbolic error despite potentially low numeric error in the training range.
*   **The Fix:** **Equation Embeddings** (as used in MMSR) alleviate this. By mapping data to a continuous embedding space, the model can place a novel OOD function "between" two known clusters (e.g., "it has properties of both sine and exponential"). The decoder can then traverse this latent space to decode a novel mix of operators, offering true combinatorial generalization rather than template classification.

## 3\. The Multi-Variate Bottleneck

The GlassBox architecture's strategy for multivariate functions ($N > 1$) involves an "ND Interpolator" followed by "1D Slicing" (fixing other variables to median).4 This heuristic is the single most significant failure point in the entire pipeline, as it relies on a mathematical assumption of **Separability** that rarely holds in complex systems.

### 3.1 Mathematical Analysis of Slicing Failure Modes

The 1D slicing strategy assumes that the multivariate function $F(x_1, \dots, x_n)$ can be reconstructed by observing its behavior along orthogonal axes passing through a central point (the median). This is formally equivalent to assuming the function is a **Generalized Additive Model (GAM)** or has a specific separable structure.

#### Failure Mode 1: The "Null Interaction"

Consider the interaction term:

$$F(x, y) = x \cdot (y - y_{med})$$

*   **Slice along X (fix $y = y_{med}$):** The term becomes $x \cdot (y_{med} - y_{med}) = 0$. The classifier observes a flat line $y=0$. It predicts "Constant" or "Identity" with coefficient 0.
*   **Slice along Y (fix $x = x_{med}$):** The term becomes $x_{med} \cdot (y - y_{med})$. The classifier observes a linear function.
*   **Result:** The classifier concludes that $F$ depends on $y$ linearly but does _not_ depend on $x$. The dependency on $x$ is effectively "masked" by the choice of the slice point. This is a catastrophic failure where a variable is completely dropped from the regression equation.

#### Failure Mode 2: Phase and Frequency Coupling

Consider a non-separable trigonometric function:

$$F(x, y) = \sin(x \cdot y)$$

*   **Slice along X (fix $y = C$):** The function is $\sin(C \cdot x)$. The classifier detects a sine wave with frequency $C$.
*   **Slice along Y (fix $x = D$):** The function is $\sin(D \cdot y)$. The classifier detects a sine wave with frequency $D$.
*   **Reconstruction:** The aggregator receives "Sine of frequency C" and "Sine of frequency D". It has absolutely no mechanism to deduce that the _true_ frequency of $x$ depends on the _value_ of $y$. It will likely propose a separable sum like $\sin(Cx) + \sin(Dy)$, which is topologically distinct from the nested product $\sin(xy)$.

### 3.2 SOTA: AI Feynman and Symmetry Breaking

The **AI Feynman** algorithm 15 treats multivariate dependencies not as a slicing problem, but as a **Symmetry Breaking** and **Decomposition** problem. This is a theoretically grounded approach based on group theory and differential geometry.

1.  **Separability Tests:** Instead of slicing, AI Feynman checks the rank of the data matrix. If we construct a matrix $M_{ij} = F(x_i, y_j)$ on a grid, the function is separable (i.e., $F(x,y) = f(x)g(y) + h(x) + k(y)$) if and only if the matrix has low rank. This is a global property of the manifold, not a local property of a slice.
2.  **Symmetry Detection:** The algorithm explicitly tests for invariances.
    *   **Translational Symmetry:** Does $F(x, y) \approx F(x + \Delta, y - \Delta)$? If so, the function depends on $(x+y)$.
    *   **Scale Symmetry:** Does $F(\lambda x, \lambda^{-1} y) \approx F(x, y)$? If so, it depends on $(xy)$.
    *   **Rotational Symmetry:** Does it depend on $x^2 + y^2$?
3.  **Recursive Decomposition:** Upon finding a symmetry (e.g., $z = xy$), the algorithm rewrites the data as $(z, F)$ and recurses. This reduces the dimensionality $N \to N-1$ in a mathematically rigorous way, preserving the interaction structure.

### 3.3 Graph Neural Networks (EvoNUDGE)

A more modern, neural-native approach to multivariate interactions is found in **EvoNUDGE**.18

*   **Graph Representation:** EvoNUDGE represents the variables $\{x_1, \dots, x_n\}$ as nodes in a fully connected graph.
*   **Message Passing:** A Graph Neural Network (GNN) processes the input data batches. The edges of the graph learn to encode the **strength of interaction** between variables.
*   **Interaction Attention:** If variables $x_i$ and $x_j$ are multiplied or divided in the ground truth formula, the GNN edge weight between them becomes high.
*   **Seed Generation:** The GNN outputs a "skeleton graph" that explicitly links interacting variables. This avoids the need for slicing entirely, as the network learns the multivariate topology directly from the joint distribution of the data.

### 3.4 Failure of Interpolation in High Dimensions

GlassBox uses an **ND Interpolator** 4 to facilitate slicing. In high dimensions ($N > 4$), data becomes sparse (the Curse of Dimensionality).

*   **Interpolation Error:** The error of any interpolator (Linear or Nearest Neighbor) grows exponentially with dimension.
*   **Ghost Artifacts:** Interpolating in sparse regions introduces "ghost" features—wiggles or plateaus that do not exist in the true function. The sensitive derivative features of GlassBox will pick up these artifacts, leading to hallucinated complexity in the symbolic regression stage.
*   **Contrast:** E2E Transformers and PointNets do not interpolate; they consume the sparse point cloud directly and learn a manifold representation that is robust to sparsity.

## 4\. ONN Warm-Start & Bias Efficiency

The GlassBox pipeline uses the classifier probabilities to **bias the logits** of the operator selectors in the ONN (Operational Neural Network) population.4 While this "Soft Prior" is better than random initialization, it is fundamentally inefficient for solving the combinatorial structure of mathematical expressions.

### 4.1 Logit Biasing: A Weak Prior for Structured Search

Logit biasing operates on the "Bag of Operators" assumption.

*   **The Mechanism:** The classifier says "Sine is likely (90%)." The ONN initializer increases the probability of choosing sin nodes during tree construction.
    *   **The Flaw:** This does not convey **structural** or **positional** information.
        *   Knowing that $\sin$ is present is only 10% of the solution.
        *   Knowing _where_ $\sin$ is (e.g., $\sin(x)$ vs. $\exp(\sin(x))$ vs. $\sin(xy)$) is the other 90%.
*   **Optimization Landscape:** Logit biasing warps the probability surface of the search, making the "sine" valley deeper. However, if the true solution requires a specific nested structure, the search algorithm still has to stumble upon that topology randomly. High operator probability can even be detrimental if the operator is used in the wrong context (e.g., spamming sin nodes everywhere, creating a "bloated" tree that is hard to optimize).

### 4.2 The "Seeding" Paradigm: PIGP

The State-of-the-Art has moved to **Hard Structural Seeding**. The **PIGP (Partially Initialized Genetic Programming)** architecture 3 demonstrates that direct injection of full solution trees is vastly superior to logit biasing.

*   **Methodology:**
    1.  A pre-trained Transformer (like the E2E model) generates $K$ candidate equations (complete trees) using Beam Search.
    2.  The Genetic Programming (GP) population is initialized with these $K$ trees (e.g., 20% of the population) plus random individuals.
*   **Why it works:**
    *   **Warm-Start vs. Hot-Start:** Logit biasing is a "Warm-Start" (the search is nudged in the right direction). Seeding is a "Hot-Start" (the search starts _at_ the likely solution).
    *   **Refinement:** The GP/ONN acts as a **local optimizer** or refiner. It takes the Transformer's guess (which might have the right structure but wrong constants, e.g., $2.1x^2$ instead of $2x^2$) and perfects it.
    *   **Diversity:** By keeping 80% of the population random, PIGP prevents premature convergence while ensuring that if the neural network was correct, the solution is found almost instantly.

### 4.3 Guided Evolution: StruSR and Masked Attribution

**StruSR (Structure-Aware Symbolic Regression)** 2 introduces a mechanism that guides the evolution _during_ the search, not just at initialization.

*   **Masking-Based Attribution:** StruSR evaluates every subtree in the population. It asks: "How much does this subtree contribute to minimizing the Physics/Derivative error?"
    *   It computes **Sensitivity Scores** for each node using gradients from a PINN surrogate.
*   **Guided Mutation:**
    *   **High Sensitivity Nodes:** (e.g., the $\sin(x)$ part that fits the data perfectly) are "protected" (frozen or given low mutation probability).
    *   **Low Sensitivity Nodes:** (e.g., a noise-fitting polynomial tail) are targeted for aggressive mutation.
*   **GlassBox Implication:** Currently, your ONN likely mutates trees randomly. This "Destructive Crossover" often breaks good partial solutions. Implementing a sensitivity-based mutation operator (guided by your classifier or a surrogate) would dramatically increase search efficiency.

### 4.4 LogicSR: Priors in MCTS

**LogicSR** 20 applies priors to **Monte Carlo Tree Search (MCTS)** rather than simple GP.

*   **Policy Network:** The classifier acts as a Policy Network $P(action|state)$.
*   **Planning:** Instead of greedy selection (like logit biasing), MCTS uses the policy to "look ahead." It explores branches that are promising according to the classifier but balances this with exploration (UCB score). This is mathematically more robust than logit biasing because it accounts for the _sequence_ of decisions required to build a valid tree.

## 5\. Specific Recommendations for Upgrade

To modernize "GlassBox" and maximize the probability of exact symbolic recovery, three high-impact, concrete architectural changes are recommended. These recommendations are prioritized by their potential to remove the identified bottlenecks.

### Recommendation 1: Replace Handcrafted Features with a Dual-Encoder Contrastive Architecture (MMSR-Style)

**Objective:** Eliminate the information loss of FFT/Derivatives and enable zero-shot structural generalization.

**Implementation Plan:**

1.  **Architecture:** Discard the 334-feature extractor. Implement a **Dual-Encoder** system:
    *   **Data Encoder:** A **Set Transformer** or **PointNet++** that ingests raw $\{(x_i, y_i)\}_{i=1}^N$ tuples. This preserves phase, topology, and supports variable input sizes.
    *   **Skeleton Encoder:** A Transformer that embeds symbolic expression trees (in Polish notation) into the same latent dimension $D=512$.
2.  **Training Objective:** Use **InfoNCE (Contrastive Loss)** to train the encoders.  
    $$\mathcal{L} = -\sum \log \frac{\exp(sim(z_{data}, z_{tree})/\tau)}{\sum_{neg} \exp(sim(z_{data}, z_{neg})/\tau)}$$  
    This aligns the numeric and symbolic manifolds.
3.  **Inference:** For a given curve, compute its data embedding $z_{data}$. Perform a **Nearest Neighbor Search** in the pre-computed library of skeleton embeddings to retrieve the top-50 most semantically similar structures. **Key Reference:** _Li et al., "MMSR: Symbolic regression is a multi-modal information fusion task", Information Fusion (2025)_.1

### Recommendation 2: Abolish 1D Slicing in Favor of Interaction-Aware Attention

**Objective:** Solve the Multivariate Bottleneck and correctly model coupled/non-separable functions.

**Implementation Plan:**

1.  **Input:** Treat the multivariate tuple $(x_1, x_2, \dots, x_n, y)$ as a single token. Feed the sequence of these tokens into a **Self-Attention Transformer**.
2.  **Interaction Map:** The internal **Attention Map** ($A \in \mathbb{R}^{N \times N}$) of the Transformer naturally encodes the coupling strength. If the function is $x_1 \cdot x_2$, the attention heads will learn to attend to both variables simultaneously.
3.  **Graph Construction:** Explicitly extract the interaction graph from the attention weights (thresholding the attention matrix). Use this graph to constrain the ONN: if $x_1$ and $x_2$ have high mutual attention, prioritize operators that combine them (like Mul(x1, x2) or Add(x1, x2)).
4.  **Symmetry Pre-check:** Before regression, run a rank-based **Separability Test** (AI Feynman style) on the data matrix to detect if the problem can be decomposed into lower-dimensional sub-problems. **Key Reference:** _Udrescu & Tegmark, "AI Feynman: A Physics-Inspired Method for Symbolic Regression", Sci. Adv. (2020)_ 15 and _EvoNUDGE_.18

### Recommendation 3: Shift from Logit Biasing to "PIGP" Structural Seeding

**Objective:** Transform the Warm-Start from a weak probabilistic nudge to a strong structural initialization.

**Implementation Plan:**

1.  **Generative Mode:** Upgrade the Classifier to an **Autoregressive Generator** (Seq2Seq). Instead of outputting class probabilities, it should output the sequence of tokens representing the formula (e.g., ["mul", "x", "sin", "y"]).
2.  **Beam Search:** At inference time, use **Beam Search** (width=50) to generate the top-50 candidate equations.
3.  **Population Seeding:** Initialize the ONN population with these 50 trees. Fill the rest of the population with random trees to maintain diversity.
4.  **Guided Mutation:** Implement **Subtree Sensitivity** (from StruSR). Before mutating a seeded tree, calculate the gradient of the loss with respect to its subtrees. Only mutate the subtrees that have high error contributions, preserving the correct structures identified by the Generator. **Key Reference:** _Jha et al., "Evolutionary and Transformer based methods for Symbolic Regression", NeurIPS 2024_.3

### Comparison of Current vs. Proposed Architecture

| Feature | Current "GlassBox" | Proposed Modernized Pipeline | Impact |
| --- | --- | --- | --- |
| Input | 334 Handcrafted Features (FFT, Derivs) | Raw Point Sets / Tokens (Set Transformer) | Captures phase, phase-coupling, and topological invariants. |
| Data Gen | Fixed Templates (Simple, Rational) | LLM Distillation / PCFG | Generalizes to OOD scientific formulas; removes template bias. |
| Multivariate | 1D Slicing + Aggregation | Self-Attention / GNN Interaction | Correctly solves non-separable functions (e.g., $\sin(xy)$). |
| Prior | Operator Logit Biasing | Structural Seeding (PIGP) | Injects full grammatical structures; exponential speedup. |
| Search | ONN (Standard Evolution) | Guided Evolution (StruSR) | Protects valid subtrees; focuses search on high-error nodes. |

### Conclusion

The "GlassBox" pipeline is a robust implementation of the **2018–2020 era** of symbolic regression. It effectively automates the intuition of a data scientist looking at a plot. However, the **2024–2025 era** is defined by **Foundation Models** that learn the manifold of mathematics itself. By replacing the manual feature extraction and slicing heuristics with **End-to-End Multimodal Encoders** and **Interaction-Aware Attention**, GlassBox can transcend its current limitations. The shift from "classifying curves" to "embedding equations" is not merely an incremental improvement; it is the necessary evolution to build a system capable of genuine scientific discovery in the age of Neuro-Symbolic AI.

#### Works cited

1.  MMSR: Symbolic Regression is a Multimodal Task - arXiv, accessed February 16, 2026, [https://arxiv.org/html/2402.18603v1](https://arxiv.org/html/2402.18603v1)
2.  (PDF) StruSR: Structure-Aware Symbolic Regression with Physics-Informed Taylor Guidance - ResearchGate, accessed February 16, 2026, [https://www.researchgate.net/publication/396330350_StruSR_Structure-Aware_Symbolic_Regression_with_Physics-Informed_Taylor_Guidance](https://www.researchgate.net/publication/396330350_StruSR_Structure-Aware_Symbolic_Regression_with_Physics-Informed_Taylor_Guidance)
3.  Evolutionary and Transformer based methods for Symbolic ..., accessed February 16, 2026, [https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_115.pdf](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_115.pdf)
4.  algoritm.md
5.  PTRNet: Global Feature and Local Feature Encoding for Point Cloud Registration - MDPI, accessed February 16, 2026, [https://www.mdpi.com/2076-3417/12/3/1741](https://www.mdpi.com/2076-3417/12/3/1741)
6.  Learning Normal Flow Directly From Event Neighborhoods - arXiv, accessed February 16, 2026, [https://arxiv.org/html/2412.11284v1](https://arxiv.org/html/2412.11284v1)
7.  End-to-end Symbolic Regression with Transformers - NeurIPS, accessed February 16, 2026, [https://papers.neurips.cc/paper_files/paper/2022/file/42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2022/file/42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf)
8.  Analyzing Generalization in Pre-Trained Symbolic Regression - arXiv, accessed February 16, 2026, [https://arxiv.org/html/2509.19849v1](https://arxiv.org/html/2509.19849v1)
9.  SRBench++ : principled benchmarking of symbolic regression with domain-expert interpretation - PMC, accessed February 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/)
10.  Call for Action: Towards the Next Generation of Symbolic Regression Benchmark - arXiv, accessed February 16, 2026, [https://arxiv.org/html/2505.03977v1](https://arxiv.org/html/2505.03977v1)
11.  MMSR: Symbolic regression is a multi-modal information fusion task - ResearchGate, accessed February 16, 2026, [https://www.researchgate.net/publication/388596061_MMSR_Symbolic_regression_is_a_multi-modal_information_fusion_task](https://www.researchgate.net/publication/388596061_MMSR_Symbolic_regression_is_a_multi-modal_information_fusion_task)
12.  [2509.19849] Analyzing Generalization in Pre-Trained Symbolic Regression - arXiv, accessed February 16, 2026, [https://arxiv.org/abs/2509.19849](https://arxiv.org/abs/2509.19849)
13.  Data2Eqn: A Fine-tuning Framework for Foundation Knowledge Transfer in Generative Model Equation Learning - arXiv, accessed February 16, 2026, [https://arxiv.org/html/2508.19487v1](https://arxiv.org/html/2508.19487v1)
14.  [Literature Review] Data-Efficient Symbolic Regression via ..., accessed February 16, 2026, [https://www.themoonlight.io/en/review/data-efficient-symbolic-regression-via-foundation-model-distillation](https://www.themoonlight.io/en/review/data-efficient-symbolic-regression-via-foundation-model-distillation)
15.  AI Feynman: A physics-inspired method for symbolic regression - PMC, accessed February 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7159912/)
16.  [1905.11481] AI Feynman: a Physics-Inspired Method for Symbolic Regression - arXiv, accessed February 16, 2026, [https://arxiv.org/abs/1905.11481](https://arxiv.org/abs/1905.11481)
17.  Contemporary Symbolic Regression Methods and their Relative ..., accessed February 16, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11074949/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11074949/)
18.  Guiding Genetic Programming with Graph Neural Networks - arXiv, accessed February 16, 2026, [https://arxiv.org/html/2411.05820v1](https://arxiv.org/html/2411.05820v1)
19.  GECCO '24 Companion: Proceedings of the Genetic and Evolutionary Computation Conference Companion - sigevo, accessed February 16, 2026, [http://www.sigevo.org/gecco-2024/toc-companion.html](http://www.sigevo.org/gecco-2024/toc-companion.html)
20.  LogicSR: prior-guided symbolic regression for gene regulatory network inference from single-cell transcriptomics data - PubMed, accessed February 16, 2026, [https://pubmed.ncbi.nlm.nih.gov/41269283/](https://pubmed.ncbi.nlm.nih.gov/41269283/)
21.  LogicSR: prior-guided symbolic regression for gene regulatory network inference from single-cell transcriptomics data | Briefings in Bioinformatics | Oxford Academic, accessed February 16, 2026, [https://academic.oup.com/bib/article/26/6/bbaf621/8339795](https://academic.oup.com/bib/article/26/6/bbaf621/8339795)