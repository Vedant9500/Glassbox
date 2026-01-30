# Operation-Based Neural Networks: A Comprehensive Research Framework for Differentiable Algorithmic Discovery

## 1\. Introduction: The Paradigm Shift from Function Approximation to Algorithmic Discovery

The contemporary landscape of artificial intelligence is dominated by the connectionist paradigm, specifically the Multi-Layer Perceptron (MLP) and its derivatives (CNNs, Transformers). In this regime, the fundamental unit of computation-the artificial neuron-is a static processing element, typically computing a weighted sum of inputs followed by a fixed non-linear activation function such as ReLU or Tanh. The "learning" in these systems is confined almost exclusively to the continuous modification of synaptic weights, which govern the magnitude of signal transmission between these fixed processing units. While this approach has proven unreasonably effective for function approximation in high-dimensional spaces , it operates fundamentally as a black-box manifold approximation engine. It does not learn the _algorithm_ or the _mathematical law_ governing the data; rather, it learns a statistical emulation of that law.

The proposal for an **Operation-Based Neural Network (ONN)**, as outlined in the foundational project notes <sup>3</sup>, represents a profound inversion of this established order. In the ONN architecture, the edges of the graph act merely as carriers of information (values), while the nodes themselves become the locus of learning, capable of selecting from a discrete library of mathematical operations (e.g., $\sin, \exp, +, \times, \log$). The objective of such a network shifts from minimizing error via weight adjustment to **discovering the optimal computation graph** that maps inputs to outputs. This moves the problem domain from continuous optimization to combinatorial search and symbolic regression, aiming to produce models that are not only accurate but sparse, interpretable, and theoretically grounded.<sup>4</sup>

This report provides an exhaustive, expert-level analysis of the theoretical and practical challenges inherent in realizing the ONN vision. We synthesize findings from diverse fields-Neural Architecture Search (NAS), Differentiable Programming, Topological Data Analysis, and Genetic Programming-to construct a rigorous framework for training these discrete-continuous hybrid systems. We address the core "blockers" identified in the project notes, specifically the non-differentiability of discrete operation selection, the handling of variable arity (inputs) for different operators, and the instability of gradient-based search in rugged loss landscapes. By integrating these insights, we propose a novel algorithmic synthesis that leverages continuous relaxation, functional interpolation, and memetic evolution to enable the discovery of "Glass Box" AI models.

### 1.1 The Fundamental Duality: Structure vs. Parameter

The defining characteristic of the ONN is the separation of **Structure** (topology and operator selection) and **Parameters** (edge constants). This dichotomy mirrors the "Nature vs. Nurture" debate in biological evolution, or more formally, the distinction between **Evo-Devo** (Evolutionary Developmental Biology) and synaptic plasticity.

- **Traditional NNs:** Structure is fixed (Nature is static); Parameters are learned (Nurture is dominant).
- **NAS / AutoML:** Structure is searched (Nature is learned); Parameters are learned (Nurture is learned).<sup>6</sup>
- **ONN:** Structure is the primary output. The "weights" (edge values) are secondary constants needed to scale inputs, much like coefficients in a physics equation ($F = ma$, where $m$ is a constant specific to an object).

The project notes correctly identify the core tension: **Backpropagation assumes continuous parameters**, but **Operation Selection is a discrete choice**.<sup>3</sup> If a node switches from computing sin(x) to x^2, the loss function undergoes a discontinuity. Gradients are undefined at the jump, and "steepest descent" becomes meaningless in the discrete domain of operator tokens. This report argues that solving this requires a bridge between the two worlds: **Continuous Relaxation**, where discrete choices are approximated by differentiable probability distributions, allowing gradients to flow into the structural decisions.

### 1.2 The Vision of the "Glass Box"

The ultimate utility of the ONN lies in its potential for **Symbolic Regression**. Unlike standard regression, which fits coefficients to a pre-specified model, or Deep Learning, which fits a massive opaque model to data, Symbolic Regression searches the space of mathematical expressions to find the model itself. The ONN is essentially a **differentiable engine for Symbolic Regression**.

If successful, such a system moves beyond "prediction" to "understanding." An ONN trained on planetary motion data should not just predict the next position of Mars (as an LSTM might); it should ideally converge to a graph structure representing Newton's Law of Universal Gravitation: $F = G \frac{m_1 m_2}{r^2}$. This requires the network to select multiplication, division, and square operations in the correct topological arrangement. The complexity of this task is combinatorial, scale-dependent, and inherently non-convex.<sup>7</sup>

## 2\. Mathematical Formalism of the Operation-Based Network

To enable rigorous analysis, we must first define the ONN in terms of Graph Theory and Differentiable Programming. We adopt a notation that unifies the discrete topology with continuous data flow.

### 2.1 The Computational Graph

Let the ONN be represented by a Directed Acyclic Graph (DAG) $G = (V, E)$.

- Let $\mathcal{O} = \{o_1, o_2, \dots, o_K\}$ be the library of available primitive operations (e.g., identity, sin, cos, exp, add, mult).
- **Nodes ($V$):** Each node $v_i \in V$ represents a computational unit. In a departure from standard NNs, $v_i$ does not have a fixed activation function. Instead, it possesses a **categorical state** $k_i \in \{1, \dots, K\}$ indicating which operation $o_{k_i} \in \mathcal{O}$ it performs.
- **Edges ($E$):** An edge $e_{ji} \in E$ carries a data vector $\mathbf{x}_{ji}$ from node $j$ to node $i$.
- **Edge Constants:** Each edge $(j, i)$ is associated with a learnable scalar constant $c_{ji} \in \mathbb{R}$. This allows the network to scale inputs (e.g., for unit conversion or coefficient fitting). The signal arriving at node $i$ from node $j$ is $c_{ji} \cdot h_j$, where $h_j$ is the output of node $j$.<sup>3</sup>

### 2.2 The "Supernode" Concept and Continuous Relaxation

To make the discrete choice of $o_{k_i}$ amenable to gradient descent, we employ the **Continuous Relaxation** technique pioneered in DARTS (Differentiable Architecture Search).<sup>9</sup> We replace the discrete selection with a weighted mixture of _all_ possible operations.

Let $\boldsymbol{\alpha}_i \in \mathbb{R}^K$ be a vector of architectural parameters (logits) for node $i$. We define the "relaxed" output of node $i$, denoted $\bar{h}_i$, as:

$$\bar{h}_i(\mathbf{x}) = \sum_{k=1}^{K} p_{i,k} \cdot o_k(\mathbf{x})$$

Where $p_{i,k}$ represents the probability of selecting operation $k$. This is typically computed via the Softmax function:

$$p_{i,k} = \frac{\exp(\alpha_{i,k} / \tau)}{\sum_{j=1}^{K} \exp(\alpha_{i,j} / \tau)}$$

Here, $\tau$ is a **temperature parameter**.

- **High Temperature ($\tau \to \infty$):** The distribution $p_i$ is uniform. The node computes the average of all operations.
- **Low Temperature ($\tau \to 0$):** The distribution approaches a one-hot vector (Dirac delta). The node effectively selects the operation with the highest $\alpha_{i,k}$.

This formulation transforms the discrete optimization problem into a continuous one. We can now compute the gradient of the loss $\mathcal{L}$ with respect to the architectural parameters $\boldsymbol{\alpha}$:

$$\frac{\partial \mathcal{L}}{\partial \alpha_{i,k}} = \sum_{\text{paths}} \frac{\partial \mathcal{L}}{\partial \bar{h}_i} \cdot \frac{\partial \bar{h}_i}{\partial p_{i,k}} \cdot \frac{\partial p_{i,k}}{\partial \alpha_{i,k}}$$

Since $\frac{\partial \bar{h}_i}{\partial p_{i,k}} = o_k(\mathbf{x})$, the gradient signal essentially asks: "If the contribution of operation $k$ (e.g., sine) were increased, would the total loss decrease?" This allows standard optimizers like SGD or Adam to "search" for the best operation by pushing $\alpha_{i,k}$ up or down.<sup>9</sup>

### 2.3 The Arity Mismatch Problem

The project notes highlight a critical blocker: **Variable Arity**.<sup>3</sup>

- Unary operators (e.g., $\sin, \cos$) require 1 input.
- Binary operators (e.g., $+$, $\times$) require 2 inputs.
- Aggregation operators (e.g., $\sum$, $\text{mean}$) can take $N$ inputs.

In a standard MLP or DARTS setup, nodes simply sum all incoming edges: $x_{\text{in}} = \sum c_{ji} h_j$. This implicitly treats every node as an aggregator. However, for an ONN to discover formulas like $y = x_1 \cdot \sin(x_2)$, the multiplication node must receive distinct inputs $x_1$ and $\sin(x_2)$, not their sum. Furthermore, division ($/$ ) is non-commutative; the order of inputs matters.

Formalizing the Solution: Differentiable Routing

To solve this, we must decouple Input Selection from Operation Execution. We propose treating the topology learning as a Routing Problem.[12]

Let us define a Routing Matrix $\mathbf{R}_i$ for each node $i$. If the maximum arity of any operation in $\mathcal{O}$ is $A_{\max}$ (typically 2 for symbolic regression), then node $i$ has $A_{\max}$ input slots.

$\mathbf{R}_i \in \mathbb{R}^{A_{\max} \times |V_{\text{prev}}|}$ contains the logits for selecting which previous node connects to which input slot.

The input to slot $s$ of node $i$ is:

$$u_{i,s} = \sum_{j \in V_{\text{prev}}} \text{Softmax}(\mathbf{R}_{i,s})_j \cdot (c_{ji} \cdot h_j)$$

The relaxed output of node $i$ then becomes a mixture over operations, where each operation utilizes the appropriate number of slots:

$$\bar{h}_i = \sum_{o_k \in \mathcal{O}_{\text{unary}}} p_{i,k} o_k(u_{i,1}) + \sum_{o_k \in \mathcal{O}_{\text{binary}}} p_{i,k} o_k(u_{i,1}, u_{i,2})$$

This formulation allows the network to learn:

- **Which operation to perform** (via $\boldsymbol{\alpha}$).
- **Which inputs to use** (via $\mathbf{R}$).
- How to scale those inputs (via $\mathbf{c}$).  
    Crucially, all three components are fully differentiable. This solves the user's "Blocker 3" regarding variable arity by explicitly modeling input slots rather than collapsing everything into a sum.[12]

## 3\. The Core Challenge: Solving Discrete Optimization in Continuous Spaces

The user notes that "Training is unstable" and Gumbel-Softmax is a "hack." This section analyzes why this instability occurs and provides rigorous mathematical solutions.

### 3.1 The Discretization Gap

The fundamental issue with the "Supernode" relaxation (Section 2.2) is the Discretization Gap.

During training, the network uses the "soft" output $\bar{h}_i$, which is a linear combination of operations (e.g., $0.5 \sin(x) + 0.5 x^2$). The optimizer finds weights that work well for this specific mixture.

However, at inference time, we must select a single discrete operation (e.g., $\sin(x)$). The behavior of $\sin(x)$ is radically different from $0.5 \sin(x) + 0.5 x^2$. The network, having overfitted to the mixture, often collapses when forced to choose. This creates a "rugged" loss landscape where the continuous minimum does not correspond to the discrete minimum.[15]

### 3.2 Advanced Relaxation Strategies

To mitigate this, we must move beyond standard Softmax and Gumbel-Softmax.

#### 3.2.1 The Hard Concrete Distribution

The Hard Concrete distribution (also known as the Rectified Gumbel-Softmax) is a rigorous improvement over the standard Gumbel-Softmax.[17]

Standard Gumbel-Softmax samples values in the open interval $(0, 1)$; it never produces exact zeros or ones. The Hard Concrete distribution stretches the underlying Gumbel distribution to the interval $(-\beta, 1+\beta)$ (where $\beta > 0$) and then clips samples to

$$via a hard sigmoid:$$

z = \text{clip}\left( \text{Sigmoid}(\log \alpha + \text{Gumbel}) \cdot (1 + 2\beta) - \beta, 0, 1 \right) $$

Why this helps: This allows the network to assign exact zero probability to certain operations during the forward pass. This means the network actually experiences the discrete removal of operations during training, rather than just attenuating them. This closes the gap between training (soft) and inference (hard).

#### 3.2.2 Entropy Regularization & Annealing

To force the network to make decisive choices, we must reshape the loss landscape. We introduce an Entropy Regularization term:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \sum_{i} H(\mathbf{p}_i)$$

where $H(\mathbf{p}_i) = -\sum_k p_{i,k} \log p_{i,k}$ is the entropy of the operation distribution at node $i$.

- Minimizing entropy forces the distribution $p_i$ toward a vertex of the simplex (a one-hot vector).
- **Annealing Strategy:** We start with high temperature $\tau$ (exploration) and zero $\lambda$. As training progresses, we decrease $\tau$ and increase $\lambda$. This allows the network to explore diverse compositions early on and then "crystallize" into a discrete architecture.<sup>15</sup>

### 3.3 Gradient Variance and Operation Scaling

Another source of instability is the vast difference in gradient magnitudes between operations.

- $o(x) = x$: Gradient is 1.
- $o(x) = \exp(x)$: Gradient is $e^x$ (can be huge).
- $o(x) = \sin(x)$: Gradient is $\cos(x)$ (bounded in $\[-1, 1\]$).

If a Supernode mixes these, the operation with the largest gradient magnitude (often exp) will dominate the learning signal, potentially starving useful but lower-magnitude operations.

Solution: Apply Batch Normalization or Layer Normalization immediately after each operation in the Supernode, before the weighted sum.[20]

$$\bar{h}_i = \sum_{k} p_{i,k} \cdot \text{BatchNorm}(o_k(u_i))$$

This normalizes the scale of the forward signals and the backward gradients, ensuring fair competition between operations.

## 4\. Novel Mathematical Approaches: Functional Interpolation

The user explicitly asked for "own maths" and new research directions, specifically asking: "Can sin and cos be smooth? Embed all operations in a continuous latent space?".[3]

This is the most promising frontier for ONNs. If we can define a continuous manifold that contains our discrete operations as special points, we can traverse this manifold using gradient descent without relaxation.

### 4.1 Fractional Hyperoperations (Interpolating **$+$** and **$\times$**)

Is there a continuous operation that slides smoothly between Addition and Multiplication?

Research into Fractional Hyperoperations suggests yes.[21]

Let us define a generalized arithmetic operator $\Phi(x, y; \beta)$ where $\beta \in $ is a continuous parameter.

- $\beta=1 \implies x+y$
- $\beta=2 \implies x \cdot y$

A simple linear interpolation (convex combination) is:

$$\Phi_{\text{linear}}(x, y; \beta) = (2-\beta)(x+y) + (\beta-1)(x \cdot y)$$

While differentiable, this doesn't capture the mathematical essence of "rank." A more rigorous approach uses Abel's Functional Equation or fractional logarithms [22], but for neural network purposes, a log-space interpolation is often numerically superior for positive inputs:

$$\Phi_{\text{log}}(x, y; \beta) = \exp\left( \log(x) + \log(y) \cdot (\beta-1) + \text{constant}(\beta) \right)$$

This concept can be extended to Neural Arithmetic Logic Units (NALU) [24], which use a gated mechanism to switch between additive and multiplicative paths. By making the gate parameter $\beta$ learnable, the network can smoothly transition a node from an "Adder" to a "Multiplier."

### 4.2 Parametric Activation Manifolds

Instead of discrete sin, cos, and tanh, we can define a single Meta-Periodic Operation:

$$o_{\text{periodic}}(x; \omega, \phi, A) = A \cdot \sin(\omega x + \phi)$$

- Learnable parameters: Frequency $\omega$, Phase $\phi$, Amplitude $A$.
- $\phi \approx 0 \implies \sin(x)$
- $\phi \approx \pi/2 \implies \cos(x)$
- $\omega \to 0 \implies \sin(\omega x) \approx \omega x$ (Linear approximation)

Similarly, for power functions (square, sqrt, identity, reciprocal):

$$o_{\text{power}}(x; p) = \text{sign}(x) \cdot |x|^p$$

- $p=1 \implies$ Identity
- $p=2 \implies$ Square
- $p=0.5 \implies$ Square Root
- $p=-1 \implies$ Reciprocal

**Research Insight:** By replacing the 29 discrete operations with just 3-4 "Meta-Operations" (Arithmetic, Periodic, Power) equipped with learnable continuous parameters, we drastically reduce the search space size and smooth the loss landscape. The problem becomes one of **parameter optimization** (well-suited for SGD) rather than topology search. After training, we can "snap" the parameters to the nearest integer/standard value (e.g., if $p=1.98$, round to 2) to recover the symbolic expression.<sup>25</sup>

### 4.3 Operator Embeddings and Hypernetworks

We can take the "embedding" idea further by using DeepONet or Neural Operator theory.[27]

Assume there exists a latent vector space $\mathcal{Z} \in \mathbb{R}^d$ where every point represents a function.

We train a Hypernetwork $\mathcal{H}: \mathbb{R}^d \times \mathbb{R} \to \mathbb{R}$.

$\mathcal{H}(\mathbf{z}, x)$ computes $f(x)$ where the function $f$ is determined by the code $\mathbf{z}$.

- Pre-training: Train $\mathcal{H}$ such that $\mathcal{H}(\mathbf{z}_{\sin}, x) \approx \sin(x)$, $\mathcal{H}(\mathbf{z}_{\text{sq}}, x) \approx x^2$, etc.
- ONN Training: Each node $i$ learns a latent vector $\mathbf{z}_i$. The output is $h_i = \mathcal{H}(\mathbf{z}_i, \text{input})$.
- **Advantage:** This maps the discrete search problem into a continuous vector space search. We can use gradient descent to move $\mathbf{z}_i$ from the "sine" region to the "square" region of the latent space.

## 5\. Optimization Strategy: The Hybrid "Memetic" Algorithm

The user asks: _"How do we design a training algorithm?"_ and suggests hybrid evolution. The research strongly supports this direction. Pure gradient descent gets stuck in local optima; pure evolution is too slow for high-dimensional parameter tuning.

### 5.1 The Limitations of Pure Gradients (Loss Landscape Analysis)

Recent studies on the loss landscapes of discrete architectures reveal they are multifractal and filled with saddle points.[29] A continuous relaxation smoothes this, but often creates spurious minima-solutions that look good in the relaxed space (averages of functions) but are invalid in the discrete space.

Gradient descent is essentially a local search method. In the space of equations, the "path" from $y=x+x$ to $y=x^2$ requires crossing a high-loss barrier (since intermediate functions like $x^{1.5}$ might perform poorly on data generated by simple addition). GD cannot easily cross these barriers.

### 5.2 The GENAS / Memetic Framework

We propose a **Bilevel Optimization** strategy, specifically a **Memetic Algorithm** (Evolution + Local Search).<sup>31</sup> This aligns with the "Lamarckian" evolution mentioned in the research snippets.<sup>34</sup>

**The Algorithm: EmbedGrad-Evolution**

- **Population Initialization:** Create a population $P$ of random ONN architectures.
- Lifetime Learning (The Gradient Step):  
    For each individual $G \in P$:
  - Freeze the topology (operations and connections).
  - Use a second-order optimizer like **L-BFGS** to optimize the continuous edge constants ($\mathbf{c}$). Note: Research shows L-BFGS is significantly better than Adam for finding constants in symbolic regression.<sup>36</sup>
  - Update the individual's fitness based on validation loss (and complexity penalty).
- **Evolutionary Selection:** Select the top $k$ individuals based on fitness.
- **Variation (The Topology Step):**
  - **Mutation:** Randomly change an operation (e.g., $\sin \to \cos$) or add/remove an edge.
  - **Crossover:** Swap sub-graphs between two high-performing individuals (e.g., taking the "multiplication" sub-module from one and the "sine" sub-module from another).
- **Gradient-Guided Mutation (Optional):** Use the gradients from a "Supernet" (Section 2.2) to bias the mutations. If the gradient $\nabla_{\alpha} \mathcal{L}$ suggests that sin is promising, increase the probability of mutating into sin.<sup>38</sup>
- **Repeat** until convergence.

This hybrid approach leverages the strength of **Evolution** for global exploration (jumping between functional forms) and **Gradients/BFGS** for local exploitation (fitting constants).

## 6\. Architecture & Implementation: Moving Beyond the RNN

The user's current implementation uses an RNN. This is likely suboptimal for general symbolic regression. RNNs impose a sequential bias (processing inputs one by one) and suffer from vanishing gradients over long sequences. Mathematical formulas are **hierarchical**, not strictly sequential (e.g., tree structures).

### 6.1 Recommended Architecture: Differentiable Cartesian Genetic Programming (dCGP)

The research highlights **Cartesian Genetic Programming (CGP)** as a superior representation for this task.<sup>40</sup>

- **Structure:** A 2D grid of nodes. Inputs enter from the left; outputs are taken from the right.
- **Connectivity:** Feed-forward connections are allowed from any previous column. This naturally encodes a DAG.
- **Gene:** Each node is defined by a gene $\[f, c_1, c_2, w_1, w_2\]$, representing the function type, input indices, and weights.
- **Differentiability:** The **dCGP** library <sup>41</sup> implements a fully differentiable version of CGP using automated differentiation (dual numbers or chain rule) to compute derivatives of outputs w.r.t inputs and weights.

Why dCGP fits the ONN vision:

It explicitly separates the discrete genotype (which function, which input index) from the continuous phenotype (weights). It supports the "Memetic" cycle perfectly: evolve the integer genes, gradient-descend the float genes.

### 6.2 Implementation Details for "EmbedGrad" Framework

To implement the **EmbedGrad** framework proposed in this report using PyTorch:

- **Define the Operator Library:** Implement MetaPeriodic, MetaPower, and MetaArithmetic modules (Section 4.2).
- **Routing Layer:** Implement the Differentiable Routing Matrix (Section 2.3) using torch.nn.Parameter and torch.softmax.
- Loss Function:  
    <br/>$$\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda_1 \|\mathbf{c}\|_1 + \lambda_2 \sum H(\mathbf{p}_i) + \lambda_3 \text{Complexity}(G)$$
  - $\|\mathbf{c}\|_1$: L1 norm on edge constants to encourage sparsity (remove weak edges).
  - $H(\mathbf{p}_i)$: Entropy regularization to force discrete op selection.
  - Complexity: Penalty for graph depth or active node count (parsimony).
- **Training Loop:**
  - Use **Adam** for the architecture parameters ($\alpha, \mathbf{R}, \omega, \phi, p$).
  - Use **L-BFGS** for the edge constants $\mathbf{c}$ in a separate fine-tuning step every $N$ epochs.

## 7\. Comparative Analysis: ONN vs. The Field

To contextualize the ONN, we compare it against established and emerging paradigms.

| **Feature** | **MLP (Traditional)** | **KAN (Kolmogorov-Arnold)** | **SymReg (Genetic Programming)** | **ONN (Proposed)** |
| --- | --- | --- | --- | --- |
| **Basic Unit** | Neuron: $\sigma(\mathbf{W}\mathbf{x})$ | Edge: Spline $\phi(x)$ | Tree Node: Operation | **Node: Operation** |
| --- | --- | --- | --- | --- |
| **Learning** | Weights (Linear) | Activation Shape | Topology & Constants | **Topology, Ops, & Constants** |
| --- | --- | --- | --- | --- |
| **Optimization** | SGD (convex-ish) | Grid Extension / SGD | Evolution (Stochastic) | **Hybrid (Grad + Evo)** |
| --- | --- | --- | --- | --- |
| **Output** | Black Box | Interpretable Univariates | Equation | **Equation (Glass Box)** |
| --- | --- | --- | --- | --- |
| **Structure** | Fixed Layers | Fixed Layers | Dynamic Tree | **Learned DAG / Grid** |
| --- | --- | --- | --- | --- |
| **Scaling** | Excellent | Moderate (slow training) | Poor (Combinatorial) | **Moderate (with Hybrid)** |
| --- | --- | --- | --- | --- |
| **Key Weakness** | Uninterpretable | Hard to find binary ops | No Gradients | **Discretization Gap** |
| --- | --- | --- | --- | --- |

Comparison with KANs:

The user specifically asked about KANs.[42] KANs are based on the theorem that multivariate functions can be decomposed into sums of univariate functions.

- **KAN:** Approximates $f(x, y) = \sin(x) + y^2$ by learning spline shapes on edges. It implicitly "learns" the operation by shaping the spline.
- **ONN:** Explicitly selects sin and square operations.
- **Advantage of ONN:** KANs struggle to represent simple interactions like multiplication ($x \times y$) compactly; they must rely on the identity $xy = \exp(\log x + \log y)$ or similar compositions. An ONN with a multiplication operator finds this instantly and exactly.

## 8\. Addressing the Specific Research Questions (Q1-Q5)

**Q1: What is the right gradient analog?**

- **Response:** The gradient is not in "operation space" but in **Selection Probability Space** ($\nabla_\alpha \mathcal{L}$) or **Continuous Parameter Space** ($\nabla_{\omega, \phi} \mathcal{L}$ for parametric ops). For the discrete jump, the "analog" is the **Forward Gradient** or **Finite Difference** approximated by the evolutionary population spread.<sup>43</sup>

**Q2: What is the right loss landscape?**

- **Response:** The raw landscape is **rugged and discontinuous**. The "right" landscape is an **Annotated Smoothing** of this. By using Meta-Ops (Section 4.2), you smooth the valleys between discrete ops. By using Entropy Regularization (Section 3.2.2), you control the "steepness" of the walls, allowing the optimizer to settle into a basin before hardening the choice.

**Q3: What is the right architecture?**

- **Response:** Not an RNN. Mathematical expressions are not time-series; they are Directed Acyclic Graphs. The optimal architecture is a **Cell-Based DAG** (like NASNet) or a **Grid** (like Cartesian GP). These structures allow for parallel branches and skip connections, which are essential for compositionality.<sup>44</sup>

**Q4: How do we handle compositionality?**

- **Response:** Compositionality is handled by the **Depth** of the graph and the **Routing**. A deep graph allows $f(g(h(x)))$. To learn this, use **Progressive Growth**: start with a shallow network and mathematically add layers (splitting nodes) only when the loss plateaus (similar to NEAT or Growing Neural Gas).<sup>46</sup>

**Q5: How do edge values interact with operations?**

- **Response:** Edge values ($c$) act as **Gain/Scaling Factors**.
  - In $y = \sin(c \cdot x)$, $c$ is the frequency.
  - In $y = c \cdot (x_1 + x_2)$, $c$ is the magnitude.
  - In $y = x^c$, $c$ is the exponent (if using a power operator).
  - Interaction: If $c \to 0$, the edge is effectively pruned. This is the primary mechanism for finding sparse structures.

## 9\. Conclusion: The Path Forward

The research confirms that "Operation-Based Neural Networks" are a viable and cutting-edge direction, residing at the frontier of Neuro-Symbolic AI. The user's intuition that "current frameworks don't work" is validated by the known failures of Gumbel-Softmax in discrete structure learning.

**Summary of the Solution Path:**

- **Abandon the "Pure" Softmax approach:** It is too unstable.
- **Adopt a Hybrid "EmbedGrad" Strategy:**
  - Use **Meta-Operations** (Parametric Sin, Power, Arithmetic) to make the local search space continuous and differentiable.
  - Use **Differentiable Routing** (Masking) to solve the arity/connection problem.
- **Use Evolution for Global Topology:** Wrap the differentiable core in a Memetic/Evolutionary loop to escape local minima.
- **Use Second-Order Optimization (BFGS):** For the edge constants, as standard SGD is too imprecise for symbolic discovery.

This framework transforms the ONN from a shaky prototype into a rigorous **Differentiable Discovery Engine**, capable of uncovering the fundamental mathematical laws hidden within data.

# Table of Contents

- **Introduction: The Paradigm Shift to Operation-Based Networks**
  - 1.1 The Duality of Structure and Parameter
  - 1.2 The "Glass Box" Vision
- **Mathematical Formalism of ONNs**
  - 2.1 Graph Definition and Topology
  - 2.2 The Supernode and Continuous Relaxation
  - 2.3 The Arity Problem and Differentiable Routing
- **Solving Technical Blockers**
  - 3.1 Discrete Optimization & The Discretization Gap (Hard Concrete vs. Softmax)
  - 3.2 Gradient Variance and Normalization
  - 3.3 Search Space & Memetic Algorithms
- **Novel Mathematical Approaches: Functional Interpolation**
  - 4.1 Fractional Hyperoperations
  - 4.2 Parametric Activation Manifolds
  - 4.3 Operator Embeddings and Hypernetworks
- **Optimization Strategy: The Hybrid Framework**
  - 5.1 Limitations of Pure Gradients
  - 5.2 The "EmbedGrad-Evolution" Algorithm
- **Architecture Recommendations**
  - 6.1 Differentiable Cartesian Genetic Programming (dCGP)
  - 6.2 Implementation Blueprint
- **Comparative Analysis (ONN vs KAN vs MLP)**
- **Addressing Research Questions (Q1-Q5)**
- **Conclusion**

#### Works cited

- operation_nn_research.txt
- Symbolic regression - Wikipedia, accessed January 21, 2026, <https://en.wikipedia.org/wiki/Symbolic_regression>
- Neural Symbolic Regression - Emergent Mind, accessed January 21, 2026, <https://www.emergentmind.com/topics/neural-symbolic-regression>
- Neural architecture search - Wikipedia, accessed January 21, 2026, <https://en.wikipedia.org/wiki/Neural_architecture_search>
- Sparse Interpretable Deep Learning with LIES Networks for Symbolic Regression, accessed January 21, 2026, <https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_283.pdf>
- Sparse Interpretable Deep Learning with LIES Networks for Symbolic Regression - arXiv, accessed January 21, 2026, <https://arxiv.org/abs/2506.08267>
- DARTS: DIFFERENTIABLE ARCHITECTURE SEARCH - OpenReview, accessed January 21, 2026, <https://openreview.net/pdf?id=S1eYHoC5FX>
- Neural Architecture Search: Two Constant Shared Weights Initialisations - arXiv, accessed January 21, 2026, <https://arxiv.org/html/2302.04406v3>
- Evolution of DARTS: A Comprehensive Guide to Neural Architecture Search - Medium, accessed January 21, 2026, <https://medium.com/@ykarray29/evolution-of-darts-a-comprehensive-guide-to-neural-architecture-search-eada9a77e01a>
- Improving River Routing Using a Differentiable Muskingum‐Cunge Model and Physics‐Informed Machine Learning - the NOAA Institutional Repository, accessed January 21, 2026, <https://repository.library.noaa.gov/view/noaa/63586/noaa_63586_DS1.pdf>
- Rich vehicle routing optimization based on variable neighborhood descent and differential evolution algorithm - PMC - PubMed Central, accessed January 21, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC12464255/>
- Differentiable Mask for Pruning Convolutional and Recurrent Networks - IEEE Xplore, accessed January 21, 2026, <https://ieeexplore.ieee.org/iel7/9106968/9108521/09108674.pdf>
- TA-DARTS: Temperature Annealing of Discrete Operator Distribution for Effective Differential Architecture Search - MDPI, accessed January 21, 2026, <https://www.mdpi.com/2076-3417/13/18/10138>
- \[2302.05629\] Improving Differentiable Architecture Search via Self-Distillation - arXiv, accessed January 21, 2026, <https://arxiv.org/abs/2302.05629>
- Just Relax It! Leveraging relaxation for discrete variables optimization | by Nikita Kiselev, accessed January 21, 2026, <https://medium.com/@kisnikser/just-relax-it-leveraging-relaxation-for-discrete-variables-optimization-717bf2bea1b8>
- The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables - Bayesian Deep Learning, accessed January 21, 2026, <http://bayesiandeeplearning.org/2016/papers/BDL_31.pdf>
- Controlling Continuous Relaxation for Combinatorial Optimization - OpenReview, accessed January 21, 2026, [https://openreview.net/forum?id=ykACV1IhjD¬eId=x9M5ZRQCV6](https://openreview.net/forum?id=ykACV1IhjD&noteId=x9M5ZRQCV6)
- EC-DARTS: Inducing Equalized and Consistent Optimization Into DARTS - CVF Open Access, accessed January 21, 2026, <https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_EC-DARTS_Inducing_Equalized_and_Consistent_Optimization_Into_DARTS_ICCV_2021_paper.pdf>
- Continuum between addition, multiplication and exponentiation? - Math Stack Exchange, accessed January 21, 2026, <https://math.stackexchange.com/questions/1269643/continuum-between-addition-multiplication-and-exponentiation>
- A DIFFERENTIABLE TRANSITION BETWEEN ADDITIVE AND MULTIPLICATIVE NEURONS - OpenReview, accessed January 21, 2026, <https://openreview.net/pdf/MwVPvKwRvsqxwkg1t7kY.pdf>
- between addition and multiplication - Tetration Forum, accessed January 21, 2026, <https://tetrationforum.org/showthread.php?tid=608>
- iNALU: Improved Neural Arithmetic Logic Unit - PMC, accessed January 21, 2026, <https://pmc.ncbi.nlm.nih.gov/articles/PMC7861275/>
- Parametric Activation Functions for Neural Networks: A Tutorial Survey - IEEE Xplore, accessed January 21, 2026, <https://ieeexplore.ieee.org/iel8/6287639/6514899/10705284.pdf>
- padé activation units: end-to-end learning - arXiv, accessed January 21, 2026, <https://arxiv.org/pdf/1907.06732>
- Neural Operators: Theory & Applications - Emergent Mind, accessed January 21, 2026, <https://www.emergentmind.com/topics/neural-operators>
- Operator Feature Neural Network for Symbolic Regression | Request PDF - ResearchGate, accessed January 21, 2026, <https://www.researchgate.net/publication/383153553_Operator_Feature_Neural_Network_for_Symbolic_Regression>
- Evaluating Loss Landscapes from a Topology Perspective - UC Berkeley Statistics, accessed January 21, 2026, <https://www.stat.berkeley.edu/~mmahoney/pubs/44_Evaluating_Loss_Landscapes_.pdf>
- \[PDF\] Evaluating Loss Landscapes from a Topology Perspective - Semantic Scholar, accessed January 21, 2026, <https://www.semanticscholar.org/paper/d36dda8bc857e6a68015559f87549063092b0390>
- A Memetic Algorithm based on Variational Autoencoder for Black-Box Discrete Optimization with Epistasis among Parameters - IEEE Xplore, accessed January 21, 2026, <https://ieeexplore.ieee.org/iel8/11042929/11042912/11042948.pdf>
- A Hybrid Genetic Algorithm + Gradient Descent Solution - Kaggle, accessed January 21, 2026, <https://www.kaggle.com/competitions/trojan-horse-hunt-in-space/writeups/2nd-place-solution-a-hybrid-genetic-algorithm-grad>
- Memetic algorithm - Wikipedia, accessed January 21, 2026, <https://en.wikipedia.org/wiki/Memetic_algorithm>
- The aim of research will be to study how through simple means of genetic algorithm search towards optimal neural network archite, accessed January 21, 2026, <https://iasj.rdd.edu.iq/journals/uploads/2024/12/16/3f1837a7004e63fa6108d17925f03a8b.pdf>
- Meta-Learning by the Baldwin Effect - arXiv, accessed January 21, 2026, <https://arxiv.org/pdf/1806.07917>
- Benchmarking symbolic regression constant optimization schemes - arXiv, accessed January 21, 2026, <https://arxiv.org/html/2412.02126v1>
- Evaluating Methods for Constant Optimization of Symbolic Regression Benchmark Problems - Computer Science, accessed January 21, 2026, <http://www.cs.mun.ca/~banzhaf/papers/BRACIS2015_Symbolic.pdf>
- A Gradient-Guided Evolutionary Neural Architecture Search | Request PDF - ResearchGate, accessed January 21, 2026, <https://www.researchgate.net/publication/378873400_A_Gradient-Guided_Evolutionary_Neural_Architecture_Search>
- A Gradient-Guided Evolutionary Neural Architecture Search - University of Surrey, accessed January 21, 2026, <https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/A-Gradient-Guided-Evolutionary-Neural-Architecture-Search/99863466602346>
- Cartesian genetic programming - Wikipedia, accessed January 21, 2026, <https://en.wikipedia.org/wiki/Cartesian_genetic_programming>
- darioizzo/dcgp: Implementation of a differentiable CGP (Cartesian Genetic Programming), accessed January 21, 2026, <https://github.com/darioizzo/dcgp>
- KAN: Kolmogorov-Arnold Networks - OpenReview, accessed January 21, 2026, <https://openreview.net/forum?id=Ozo7qJ5vZi>
- Papers Simplified: »Gradients without Backpropagation« | Towards Data Science, accessed January 21, 2026, <https://towardsdatascience.com/papers-simplified-gradients-without-backpropagation-96e8533943fc/>
- Neural Architecture Search Benchmarks: Insights and Survey - IEEE Xplore, accessed January 21, 2026, <https://ieeexplore.ieee.org/ielaam/6287639/10005208/10063950-aam.pdf>
- (PDF) Differentiable Genetic Programming - ResearchGate, accessed January 21, 2026, <https://www.researchgate.net/publication/315065599_Differentiable_Genetic_Programming>
- Growing Neural Networks: Dynamic Evolution through Gradient Descent - arXiv, accessed January 21, 2026, <https://arxiv.org/html/2501.18012v2>
- Growing neural networks: dynamic evolution through gradient descent | Proceedings A | The Royal Society, accessed January 21, 2026, <https://royalsocietypublishing.org/rspa/article/481/2318/20250222/234342/Growing-neural-networks-dynamic-evolution-through>