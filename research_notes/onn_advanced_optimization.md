# Architectural Audit: Overcoming Performance Ceilings in Neuro-Symbolic Operator Networks

> Note: This is a research note. It may discuss methods that are not implemented in the current Glassbox codebase.

## 1\. Introduction: The Neuro-Symbolic Convergence and the Optimization Cliff

The pursuit of Artificial General Intelligence has increasingly converged upon the intersection of connectionist deep learning and symbolic reasoning. This domain, often termed Neuro-Symbolic AI, seeks to combine the differentiable learnability of neural networks with the interpretability, generalization, and extrapolation capabilities of symbolic logic and arithmetic. A central artifact in this domain is the Operator Neural Network (ONN), a class of architectures designed to learn explicit mathematical operations—such as addition, multiplication, and trigonometric functions—rather than merely approximating them via high-dimensional manifolds. The ultimate goal of an ONN is not just prediction, but "symbolic discovery" or "equation mining," where the trained weights can be discretized back into a human-readable physical law or mathematical formula.

However, the transition from continuous differentiable optimization to discrete symbolic structure is fraught with pathological difficulties. Current state-of-the-art methods, particularly those relying on Differentiable Architecture Search (DARTS) and its variants, frequently encounter a "Performance Ceiling." This ceiling is characterized by a failure to improve beyond a certain sub-optimal bound, often driven by the inability of the network to navigate the optimization landscape effectively. Two primary culprits have been identified: the **Discretization Gap**, which refers to the loss of performance when the continuous "supernet" is projected onto a discrete architecture, and **Entropy Collapse**, a phenomenon where the search algorithm prematurely converges to degenerate solutions, typically favoring simple identity mappings over complex computational operators.

This architectural audit provides an exhaustive analysis of the ONN framework, diagnosing the root causes of these failures and proposing robust architectural and algorithmic remedies. The report scrutinizes the optimization dynamics of standard gradient-based search, revealing the "unfair advantage" that drives entropy collapse. It contrasts the traditional node-centric ONN (exemplified by the Neural Arithmetic Logic Unit or NALU) with the emerging edge-centric Kolmogorov-Arnold Networks (KANs), highlighting the superior gradient stability of spline-based activation functions. Furthermore, the report investigates the topology of the loss landscape itself, identifying specific saddle points and "gradient starvation" issues inherent in meta-arithmetic interpolation. Finally, it explores advanced search strategies beyond the conventional Gumbel-Softmax relaxation—such as Risk-Seeking Policy Gradients and Lamarckian Evolutionary strategies—and addresses the practical engineering challenges of deploying these dynamic, recursive graphs using modern compiler stacks like torch.compile.

The synthesis of these insights suggests that overcoming the performance ceiling requires a fundamental rethinking of how discreteness is handled in differentiable systems. It demands a shift from "exclusive competition" to "collaborative learning" in architecture search, the adoption of locally supported activation functions to mitigate catastrophic interference, and the implementation of rigorous regularization schemes that respect the scale-invariant nature of sparsity.

## 2\. The Discretization Gap and Entropy Collapse in Differentiable Search

The prevailing paradigm for discovering optimal ONN architectures has been Differentiable Architecture Search (DARTS). By relaxing the discrete search space into a continuous one, DARTS allows the use of standard gradient descent to optimize architecture parameters ($\alpha$) alongside model weights ($w$). While theoretically elegant, this relaxation introduces a severe misalignment between the training objective and the inference constraint, leading to the Discretization Gap.

### 2.1 The Mechanics of the Discretization Gap

In a typical ONN supernet, the output of a specific edge is computed as a weighted sum of all candidate operations (e.g., add, multiply, sine, identity). This is formally expressed as a Softmax-weighted mixture:

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} o(x)$$

During the search phase, the network learns to utilize this ensemble. A convolution operation might extract features, while a parallel skip connection preserves high-frequency information and gradient magnitude. The weights $w$ are optimized under the assumption that _both_ paths exist. However, at the conclusion of the search, the architecture must be discretized. The standard approach is to apply an argmax to the architecture parameters $\alpha$, retaining only the single operation with the highest weight and discarding the rest.

This projection from the continuous simplex to a one-hot vector is the source of the Discretization Gap. The removal of the "cooperative" operations creates a shock to the system. The weights $w$ of the retained operation were trained in the presence of the discarded operations and often rely on them for error correction or feature augmentation. Consequently, the performance of the discrete architecture drops precipitously compared to the continuous supernet. This gap is not merely a loss of accuracy; it fundamentally hinders real-world deployment, as the ranking of architectures in the continuous space correlates poorly with their ranking in the discrete space. The optimization trajectory effectively hallucinates a solution that cannot exist under the constraints of the final hardware or symbolic requirement.

### 2.2 Entropy Collapse: The "Unfair Advantage" Pathology

Compounding the discretization gap is the phenomenon of Entropy Collapse. Ideally, the distribution of $\alpha$ values would remain high-entropy (uncertain) during the early exploration phase and gradually sharpen as the network gathers evidence. In practice, DARTS suffers from a premature collapse where the distribution spikes towards specific operations long before the model weights $w$ have converged.

#### 2.2.1 The Hegemony of Skip Connections

The most common manifestation of entropy collapse in image-based or recurrent tasks is the dominance of skip connections (identity mappings). Research indicates that this is not due to the intrinsic superiority of skip connections for the task, but rather an "unfair advantage" in the optimization dynamics.

When the supernet is initialized, parameterized operations like convolutions or arithmetic layers (e.g., Linear, Conv2d, NALU) have random weights. They produce noise and obscure the gradient signal, especially in deep networks. Skip connections, being parameter-free, provide a direct path for gradient backpropagation, resembling the mechanics of ResNets. In the early epochs, the path of least resistance for minimizing the loss is simply to forward the input directly to the output.

The Softmax function used in standard DARTS enforces an "Exclusive Competition." Because the sum of weights must equal 1, increasing the architecture weight $\alpha_{skip}$ for the skip connection mathematically _requires_ decreasing the weights for all other operations.

$$\frac{\partial \mathcal{L}}{\partial \alpha_{conv}} \approx \text{negative, because } \alpha_{skip} \text{ is increasing}$$

This creates a "Rich-Get-Richer" feedback loop. The optimizer increases $\alpha_{skip}$ to improve gradient flow; this suppresses $\alpha_{conv}$; the convolution layer receives smaller gradient updates because its weight in the weighted sum is reduced; the convolution fails to learn useful features; the model relies even more on the skip connection. The system collapses into a monopoly state where the final architecture is a degenerate, shallow network composed almost entirely of identity mappings. This collapse is a primary driver of the "Performance Ceiling," as the network effectively prunes away its own capacity to learn complex operators.

### 2.3 Architectural Remedies: From Exclusive to Collaborative Competition

To break the performance ceiling, we must structurally alter the competition dynamics. The solution lies in decoupling the optimization of different operations and enforcing discreteness through auxiliary losses rather than rigid structural constraints like Softmax.

#### 2.3.1 FairDARTS and the Sigmoid Relaxation

The FairDARTS framework proposes replacing the Softmax activation with a Sigmoid activation, transitioning from exclusive to **Collaborative Competition**. In this regime, the weight of an operation is defined as $\sigma(\alpha_o)$, where $\sigma$ is the logistic sigmoid function.

$$\sigma(\alpha) = \frac{1}{1 + e^{-\alpha}}$$

Crucially, this allows multiple operations to be active simultaneously without suppressing one another. If the skip connection is useful, its weight can approach 1. If the convolution becomes useful later in training, its weight can _also_ approach 1. This breaks the zero-sum game that leads to the monopoly of skip connections. The "unfair advantage" is neutralized because the success of the identity path does not structurally penalize the learning rate of the parameterized path.

#### 2.3.2 The Zero-One Loss ($L_{0-1}$)

While Sigmoid allows collaborative learning, it does not inherently solve the discretization gap; in fact, it might exacerbate it by allowing the network to use _all_ operations, making the final pruning even more destructive. To counter this, FairDARTS introduces a **Zero-One Loss** term ($L_{0-1}$) designed to force the continuous weights towards the binary extremes of 0 and 1.

$$L_{0-1} = -\frac{1}{N} \sum (\sigma(\alpha) - 0.5)^2$$

_or effectively minimizing the distance to the nearest integer._

This auxiliary loss acts as a differentiable pressure towards discreteness. By penalizing values in the ambiguous range (e.g., 0.3 to 0.7), it ensures that by the end of the search, the network has naturally discretized itself. A sensitivity weight of $w_{0-1} \approx 10$ has been empirically found to balance the primary task loss and the discretization objective. This "soft" discretization allows the weights $w$ to adapt to the sparse structure gradually, rather than suffering a sudden shock at the end.

#### 2.3.3 Beta-Decay Regularization

An alternative approach to stabilizing the search is **Beta-Decay Regularization**. This method imposes constraints on the magnitude and variance of the architecture parameters $\alpha$. By penalizing the norm of activated parameters, Beta-Decay prevents any single operation from growing too dominant too quickly. This acts as a "speed limit" on entropy collapse, forcing the network to maintain a high-entropy exploration state for longer. Theoretical analysis suggests that Beta-Decay improves the transferability of architectures by preventing overfitting to the specific noise patterns of the proxy dataset used during search.

#### 2.3.4 Implicit Hessian Regularization via Gumbel-Matching

Another potent strategy for closing the discretization gap is **Gumbel-Matching**. This technique involves injecting Gumbel noise into the logits of the architecture parameters before applying the Softmax (or Sigmoid).

$$\alpha'_{o} = \alpha_o + g_o, \quad g_o \sim \text{Gumbel}(0, 1)$$

Using the Straight-Through Estimator (STE) allows gradients to flow through this stochastic process. The theoretical insight here is that the noise injection acts as an **Implicit Hessian Regularization**. It forces the optimization to find minima that are "flat" (low curvature). In a sharp minimum, a small change in architecture weights (like the shift from 0.9 to 1.0 during discretization) can cause a massive jump in loss. In a flat minimum, the loss surface is robust to these perturbations. Gumbel-Matching has been shown to reduce the discretization gap by up to 98% and effectively eliminate unused gates, leading to a significant speedup in training wall-clock time.

**Table 1: Comparison of Discretization Strategies**

| Method | Relaxation | Competition Type | Mechanism for Gap Reduction | Mechanism for Collapse Prevention |
| --- | --- | --- | --- | --- |
| DARTS | Softmax | Exclusive | None (Argmax shock) | None (Prone to collapse) |
| FairDARTS | Sigmoid | Collaborative | Zero-One Loss ($L_{0-1}$) | Decoupled weights (No monopoly) |
| Gumbel-Matching | Gumbel-Softmax | Exclusive/Hybrid | Implicit Hessian Regularization | Noise injection flattens minima |
| Beta-Decay | Softmax | Exclusive | N/A | Regularizes magnitude/variance of $\alpha$ |

## 3\. Gradient Stability in Neuro-Symbolic Architectures: ONN vs. KANs

In the domain of symbolic regression and operator learning, the choice of the fundamental building block is as critical as the search strategy. The standard Operator Neural Network (ONN) paradigm relies on node-centric operations (e.g., NALU). A rising challenger is the Kolmogorov-Arnold Network (KAN), which relies on edge-centric learnable activations.

### 3.1 The ONN Paradigm: Fragility in Log-Space

The Neural Arithmetic Logic Unit (NALU) and its predecessor, the Neural Accumulator (NAC), attempt to learn arithmetic operations by parameterizing the weight matrices to represent discrete values $\{-1, 0, 1\}$. The NAC uses a transformation $W = \tanh(\hat{W}) \odot \sigma(\hat{M})$ to accumulate inputs additively. To achieve multiplication, the NALU lifts the inputs into log-space:

$$y = \exp( W(\log(|x| + \epsilon)) )$$

While theoretically capable of representing $x_1 \cdot x_2$ (as $\exp(\log x_1 + \log x_2)$), this architecture suffers from severe gradient instability.

**1\. The Log-Space Singularity:** The reliance on logarithms makes the network fundamentally incapable of handling mixed-sign data naturally. The trick of using $|x|$ and learning a separate sign path is brittle and often fails to converge because the sign information is decoupled from the magnitude gradient. **2\. Gate Ambiguity:** NALU uses a learned gate $g$ to switch between the additive (NAC) and multiplicative (log-space) paths. The gradient of this gate often vanishes, or the gate converges to 0.5, resulting in a hybrid operation (half-add, half-multiply) that has no physical meaning and high error. **3\. Gradient Starvation:** The gradients through the multiplicative path scale with the inverse of the input ($1/x$), while additive gradients are constant. For inputs near zero, the multiplicative gradients explode; for large inputs, they vanish. This imbalance leads to "Gradient Starvation," where the stable additive path dominates the learning dynamics even if the underlying relationship is multiplicative.

### 3.2 Kolmogorov-Arnold Networks (KANs): A Spline-Based Revolution

Kolmogorov-Arnold Networks (KANs) are based on the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as a superposition of univariate continuous functions. Unlike MLPs or ONNs, which place fixed non-linearities at nodes, **KANs place learnable activation functions on edges**.

#### 3.2.1 B-Splines and Local Support

The key innovation in KANs is the use of B-splines to parameterize these univariate edge functions. A function $\phi(x)$ on an edge is learned as:

$$\phi(x) = \sum_i c_i B_i(x)$$

where $B_i(x)$ are basis functions and $c_i$ are learnable coefficients.

**Gradient Stability Advantage:** The primary advantage of B-splines regarding gradient stability is **Local Support**. In a standard MLP or ONN with global activation functions (like Sigmoid or Tanh), adjusting a weight to fix an error for input $x=5$ can disastrously impact the prediction for input $x=-5$ ("Catastrophic Interference"). In a B-spline, modifying a coefficient $c_i$ only affects the function in the narrow interval where basis function $B_i(x)$ is non-zero. This locality creates a well-conditioned optimization landscape. The gradients are decoupled across the input domain, allowing the network to fine-tune different regions of the function independently. This is particularly vital for symbolic regression, where the "shape" of the function (e.g., a sudden asymptotic rise in $1/x$) requires precise local control that global weights struggle to capture without destabilizing the rest of the manifold.

#### 3.2.2 Fast-KAN and Computational Efficiency

A criticism of spline-based networks is computational cost. Evaluating splines on a grid is slower than simple matrix multiplication. The **Fast-KAN** approach addresses this by replacing the recursive B-spline evaluation with a combination of a fast coarse-grained basis (like RBFs or simplified polynomials) and a residual fine-grained grid. This retains the gradient benefits of local adaptability while approaching the inference speed of standard MLPs, removing the computational bottleneck that previously hindered the scaling of such architectures.

### 3.3 Symbolification: Post-Hoc vs. Online

The ultimate test of these architectures is their ability to yield interpretable formulas.

**ONN Discretization:**

ONNs rely on a post-training discretization where gates are snapped to 0 or 1. As discussed, the Discretization Gap often renders the resulting formula inaccurate.

**KAN Symbolification:** KANs enable a more robust "Symbolification" process. Since every edge learns a univariate function $y = \phi(x)$, we can inspect these functions individually.

*   **Visual Inspection:** We can plot $\phi(x)$ and visually recognize it as a parabola ($x^2$), a sinusoid ($\sin x$), or an exponential.
*   **Automatic Snapping (fix_symbolic):** KANs support an interactive or automated refinement mode. If an edge function correlates highly with a symbolic candidate (e.g., $R^2 > 0.99$ for $\sin(x)$), we can replace the spline with the exact symbolic function $\sin(x)$ and freeze it.
*   **Online Refinement:** Unlike ONN's global snap, KAN allows for partial symbolification. We can fix one layer to symbolic functions and continue training the remaining spline layers to compensate for the residual error. This iterative "locking in" of the equation avoids the catastrophic performance drop of global discretization.

## 4\. Meta-Arithmetic Optimization: Saddle Points and Gradient Starvation

When an ONN attempts to learn a "Meta-Arithmetic" operation—interpolating between addition and multiplication to discover the correct operator—it traverses a loss landscape replete with specific topological hazards.

### 4.1 The Add-Multiply Interpolation Landscape

Consider a learnable unit $y = \beta(x_1 + x_2) + (1-\beta)(x_1 \cdot x_2)$. The mixing parameter $\beta$ is optimized via gradient descent. Theoretical analysis of this landscape reveals the presence of **Saddle Points** at the transition boundaries.

If the true function is multiplicative ($y=x_1 x_2$) but the network initializes in the additive regime ($\beta \approx 1$), the gradient for the multiplicative component might be obscured. The transition region often behaves as a saddle point where the curvature (Hessian) has mixed signs.

*   **The Strict Saddle Property:** Fortunately, many of these saddle points satisfy the "Strict Saddle" property, meaning there is at least one direction of negative curvature (an escape route).
*   **Escape Mechanisms:** Standard Gradient Descent can stall at saddle points for exponential time. However, algorithms that inject isotropic noise (like Perturbed SGD or zeroth-order methods with randomized smoothing) are theoretically guaranteed to escape strict saddle points in polynomial time. This validates the use of stochastic search methods (like Gumbel-Matching or Langevin dynamics) over pure deterministic SGD for ONN training.

### 4.2 Gradient Starvation and Dominance

A more pervasive issue is **Gradient Dominance** (often termed Gradient Starvation in this context). This occurs when different competing operations produce gradients of vastly different magnitudes.

In an ONN, the magnitude of the gradient through a multiplication gate depends on the input values: $\nabla_{x_1} (x_1 x_2) = x_2$. If the input features have a large variance or non-normalized scale, the multiplicative path can generate gradients that are orders of magnitude larger than the additive path (where $\nabla = 1$).

*   **The Starvation Mechanism:** In a competitive Softmax/Sigmoid setup, the optimizer will follow the largest gradient. If the multiplicative path provides a "louder" signal (even if it's just fitting noise or transient correlations), the optimizer will rapidly increase its weight, effectively "starving" the additive path before it can demonstrate its utility.
*   **Spectral Decoupling:** To mitigate this, one must employ Spectral Regularization or specific normalization schemes (like LayerNorm applied _inside_ the operator branches) to ensure that the gradient norms of candidate operations are comparable. "Gradient Starvation" theory suggests that decoupling the learning dynamics of features (or operators) is essential to prevent the dominant high-frequency features from suppressing the learning of robust low-frequency laws.

## 5\. Advanced Search Strategies: Beyond Gumbel-Softmax

While Gumbel-Softmax is the standard for differentiable discretization, it is inherently a local search method limited by the variance of the gradient estimator. For complex symbolic spaces, more aggressive or global search strategies are required.

### 5.1 Deep Symbolic Regression (DSR) and Risk-Seeking Policy Gradients

**Deep Symbolic Regression (DSR)** abandons the "supernet" relaxation entirely in favor of a generative approach. An Autoregressive Controller (typically an RNN or Transformer) generates a sequence of tokens representing a mathematical expression in Polish notation (e.g., \[+, \*, x, y, 2\]).

**The Failure of Expected Reward:**

Standard Reinforcement Learning (REINFORCE) optimizes the _expected_ reward:

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}$$

This creates a perverse incentive in symbolic regression. A policy that generates one perfect formula ($R=1.0$) and 99 invalid ones ($R=0.0$) has a mean reward of 0.01. A policy that generates 100 mediocre approximations ($R=0.5$) has a mean reward of 0.5. Standard RL would prefer the mediocre policy. However, in scientific discovery, we only care about the _single best_ formula found; the failed attempts are irrelevant.

**Risk-Seeking Policy Gradient:** DSR addresses this by optimizing for the "Best-Case" scenario using a **Risk-Seeking Policy Gradient**.

$$\nabla J_{risk} \approx \frac{1}{| \mathcal{T}_{top} |} \sum_{\tau \in \mathcal{T}_{top}} (R(\tau) - b) \nabla \log p_\theta(\tau)$$

Instead of updating the controller based on the entire batch of sampled expressions, DSR filters the batch to retain only the top $k$-percentile (e.g., top 5%) of expressions based on their fitness (reward). The gradient is computed solely from these high-performers.

*   **Impact:** This artificially increases the variance of the search but aligns the optimization with the true goal. It encourages the controller to commit to "risky" patterns that yielded high rewards in the past, effectively performing a "hill-climbing" operation on the upper envelope of the reward distribution rather than the mean.

### 5.2 Lamarckian Evolution: Gradient-Informed Evolutionary Strategies

Pure evolutionary algorithms (Genetic Programming) struggle with optimization of constants (e.g., finding $3.14$ in $3.14 x^2$). Pure gradient search struggles with topology (finding $x^2$). **Lamarckian Evolution** hybridizes these by allowing individuals to "learn" during their lifetime.

**The Algorithm:**

1.  **Genotype:** A symbolic expression tree (topology).
2.  **Phenotype:** The instantiated formula with specific constants.
3.  **Local Search (Learning):** Before evaluating fitness, we perform a gradient-based optimization (e.g., BFGS or SGD) on the _constants_ of the expression to fit the data.
4.  **Write-Back:** Crucially, the optimized constants are written back into the genotype. The individual that reproduces is the _optimized_ version.

This "Lamarckian" mechanism (inheritance of acquired characteristics) smooths the fitness landscape. A correct topology with wrong constants (e.g., $1.0 x^2$) might have poor fitness initially. Local search reveals its true potential ($3.14 x^2$), giving it high fitness. This effectively decouples the discrete search for structure (handled by evolution) from the continuous search for parameters (handled by gradient descent), bridging the discretization gap by ensuring that every structural candidate is evaluated at its _best possible_ parameterization.

### 5.3 Hoyer Regularization: Scale-Invariant Sparsity

For both ONN and KAN, enforcing sparsity is essential for interpretability. Standard $L_1$ regularization is scale-dependent: shrinking the weights reduces the $L_1$ norm without necessarily setting them to zero. **Hoyer Regularization** (or $L_1/L_2$ regularization) provides a scale-invariant alternative.

$$H(w) = \frac{ \|w\|_1 }{ \|w\|_2 }$$

The Hoyer measure reflects the "peakedness" of the weight vector.

*   **Mechanism:** Minimizing $H(w)$ encourages the mass of the vector to concentrate in as few elements as possible. Unlike $L_1$, scaling the vector $w \to \lambda w$ does not change $H(w)$.
*   **Result:** This forces the optimizer to drive non-essential weights to exactly zero to minimize the ratio, rather than just shrinking them. In ONN architecture search, applying Hoyer regularization to the $\alpha$ vector results in sharper decisions and a smaller discretization gap compared to Lasso ($L_1$).

### 5.4 Gate Regularization

In architectures with explicit gating mechanisms (like Gated Continuous Logic Networks), **Gate Regularization** is employed to force gates to be binary.

$$L_{gate} = \sum_i \min(g_i, 1-g_i) \quad \text{or} \quad \sum_i g_i(1-g_i)$$

This loss is minimized when $g_i \in \{0, 1\}$. It is critical for ensuring that the logic learned by the network is Boolean and not fuzzy. Without this, a gate value of $0.5$ represents a logical ambiguity that destroys the symbolic validity of the extracted rule.

## 6\. Engineering Optimizations: Dynamic Graphs and torch.compile

The implementation of advanced ONN and KAN architectures often involves dynamic structures—recursion in DSR's controller, data-dependent control flow in gated units, and conditional execution paths. Training these efficiently on modern hardware requires navigating the constraints of the torch.compile stack (PyTorch 2.0).

### 6.1 The Challenge of Dynamic Graphs

torch.compile uses **TorchDynamo** to trace Python code into an FX graph and **TorchInductor** to compile that graph into optimized Triton kernels. The compiler relies on "Guards" to verify that assumptions made during tracing (e.g., tensor shapes, types) hold true at runtime.

**Graph Breaks:**

A "Graph Break" occurs when Dynamo encounters code it cannot trace, such as:

1.  **Data-Dependent Control Flow:** if x.sum() > 0: (The branch depends on a tensor value, unknown at compile time).
2.  **Non-Torch Libraries:** Calling numpy or scipy functions.
3.  **Dynamic Recursion:** Recursive calls with depths that vary based on data.

When a graph break occurs, the execution falls back to the slow Python interpreter, creating overhead and preventing whole-graph optimizations like kernel fusion.

### 6.2 Optimization Strategies

To leverage torch.compile for ONN/KAN, we must refactor the code to be "compiler-friendly."

#### 6.2.1 From Branching to Masking

The most common issue in ONNs is gating: if gate > 0.5: run_op1() else: run_op2().

*   **Problem:** This causes a graph break because the path is data-dependent.
*   **Solution:** Use **Masking** or torch.where.
*   Python

1.  \# Compiler Friendly
2.  res1 = run_op1(x)
3.  res2 = run_op2(x)
4.  output = torch.where(gate > 0.5, res1, res2)

*   While this appears to do more work (computing both branches), torch.compile can often fuse these operations into a single kernel, avoiding the massive overhead of CPU-GPU synchronization required to check the if condition on the host.

#### 6.2.2 Handling Recursion in Symbolic Parsers

DSR and Tree-LSTMs rely on recursion to parse or generate expression trees.

*   **Problem:** Dynamo unrolls recursion. If the recursion depth is large or variable, this leads to massive instruction traces and compilation timeouts ("Recursion Limit Hit").
*   **Solution:** **Iterative Refactoring**. Rewrite recursive tree traversals as iterative loops using explicit stacks.
*   Python

1.  \# Recursive (Bad for Compile)
2.  def traverse(node):
3.  return process(node) + traverse(node.left)
4.  \# Iterative (Good for Compile)
5.  stack = \[root\]
6.  while stack:
7.  node = stack.pop()
8.  #... process...

*   Additionally, utilizing torch.compile(dynamic=True) is essential. This tells the compiler to generate kernels that handle symbolic dimensions for shapes, preventing recompilation every time the batch size or sequence length changes—a frequent occurrence in variable-length symbolic regression tasks.

#### 6.2.3 Structured Sparsity for KANs

While Hoyer regularization induces sparsity, standard GPUs do not accelerate unstructured sparse tensors efficiently.

*   **Block Sparsity:** For KANs, it is more efficient to prune entire spline edges (structured sparsity) rather than individual coefficients. By setting the FullGraph=True flag during compilation, one can verify that the graph is free of breaks, ensuring that the pruned structure is compiled into a highly optimized, dense-like kernel that skips the zero-computations entirely.

## 7\. Conclusion

The "Performance Ceiling" in Operator Neural Networks is not an impenetrable barrier but a symptom of specific, identifiable pathologies in optimization and architecture. The **Discretization Gap** arises from the shock of projection; **Entropy Collapse** arises from the unfair advantage of identity mappings in exclusive competitions; and **Gradient Starvation** arises from the scale imbalance between arithmetic operators.

To transcend these limitations, the architectural audit prescribes a holistic set of interventions:

1.  **Shift to Collaborative Competition:** Replace Softmax with Sigmoid (FairDARTS) to allow operations to coexist, and use Zero-One Loss to induce "soft" discreteness gradually.
2.  **Adopt Edge-Centric Architectures:** Transition from NALU-style nodes to KAN-style spline edges to benefit from local support and avoid log-space singularities.
3.  **Globalize the Search:** Move beyond local Gumbel-Softmax to Risk-Seeking Policy Gradients (DSR) or Lamarckian Evolutionary strategies to escape local optima and saddle points.
4.  **Regularize for Topology:** Employ Hoyer Regularization for scale-invariant sparsity and Gate Regularization for binary logic consistency.
5.  **Compile for Dynamics:** Refactor dynamic control flows into masked operations and iterative loops to unlock the speed of torch.compile, enabling the massive iteration counts required for symbolic discovery.

By synthesizing these advanced differentiable programming techniques with robust symbolic search strategies, we can construct Neuro-Symbolic systems that do not merely approximate the world, but successfully distill its governing laws.

### Table 2: Summary of Architectural Pathologies and Solutions

| Pathology | Mechanism | Architectural Solution | Algorithmic Solution |
| --- | --- | --- | --- |
| Discretization Gap | Mismatch between soft supernet and hard architecture. | FairDARTS (Sigmoid + Zero-One Loss). | Gumbel-Matching (Implicit Hessian Reg). |
| Entropy Collapse | Unfair advantage of skip connections in exclusive Softmax. | Collaborative Competition (Sigmoid). | Beta-Decay Regularization. |
| Gradient Instability | Log-space singularities; global interference in activations. | Kolmogorov-Arnold Networks (B-Splines). | Fast-KAN (Grid interpolation). |
| Saddle Points | Mixed curvature at Add/Mult transition. | N/A | Risk-Seeking Policy Gradient; Noise Injection. |
| Gradient Starvation | Multiplicative gradients dominate additive ones. | Spectral Normalization / LayerNorm in branches. | Gradient Clipping; Feature Decoupling. |
| Graph Breaks | Dynamic control flow halts compiler tracing. | Masking (torch.where); Iterative Stacks. | torch.compile(dynamic=True). |