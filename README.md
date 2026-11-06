# neuro-sync

`neuro-sync` is a distributed PyTorch runtime framework that establishes a bidirectional neural feedback loop between independent Large Language Models. By intercepting intermediate activation manifolds and cross-feeding gradient tensors via asynchronous inter-process communication, it forces real-time weight co-adaptation—allowing two distinct models to operate as a unified cognitive topology.

---

## Architectural Mechanics

```
┌─────────────────────────┐               ┌─────────────────────────┐
│     Model A (Worker)    │               │     Model B (Worker)    │
│  [Forward Pass Tensor]  │               │  [Forward Pass Tensor]  │
└────────────┬────────────┘               └────────────▲────────────┘
             │                                         │
             ▼                                         │
┌─────────────────────────┐               ┌─────────────────────────┐
│   Async Hook Intercept  │──────────────>│    Gradient Injection   │
│  (Activation Manifold)  │  Shared IPC   │   (Latent Adaptation)   │
└─────────────────────────┘               └─────────────────────────┘

```

### 1. Non-Blocking Activation Intercept (`hooks.py`)

Uses custom PyTorch `torch.nn.Module` forward and backward hooks to capture hidden states (`hidden_states[-1]`) at specific transformer layer boundaries without stalling the host inference graph. Tensors are serialized via shared CUDA memory or zero-copy IPC ring buffers.

### 2. Cross-Model Gradient Cross-Feeding (`bridge.py`)

During backpropagation or continuous alignment passes, Model A’s loss gradients ($\frac{\partial L}{\partial a}$) are transformed through an orthogonal projection matrix and added directly to Model B’s active gradient accumulator (`param.grad.data.add_(...)`), synchronizing their optimization trajectories.

### 3. Shared Latent Adapter Mesh (`mesh.py`)

Rather than synchronizing full monolithic weight matrices, the framework maintains an out-of-core memory-mapped (`mmap`) tensor cache for Low-Rank Adaptation ($LoRA$) weights ($A$ and $B$ matrices). Updates computed by one model instantly mutate the shared projection subspace of the other.

---

## Directory Structure

```text
neuro-sync/
├── core/
│   ├── synapse.py       # Distributed process group binding & orchestration
│   └── bridge.py        # Tensor cross-feeding & gradient injection logic
├── hooks/
│   ├── intercept.py     # Non-blocking PyTorch forward/backward hooks
│   └── transform.py     # Latent space alignment & orthogonal projection
├── mesh/
│   ├── mmap_adapter.py  # Shared memory-mapped LoRA matrix management
│   └── plasticity.py    # Hebbian weight bounds & homeostasis clamping
└── runtime/
    └── worker.py        # Concurrent execution loop & synchronization clocks

```

---

## Implementation Example

```python
import torch
from neurosync import SynapseOrchestrator, SynapseConfig
from transformers import AutoModelForCausalLM

# Initialize distinct model runtimes on isolated devices
model_a = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B").to("cuda:0")
model_b = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to("cuda:1")

# Configure bidirectional gradient bridge and shared LoRA subspace
config = SynapseConfig(
    coupling_mode="bidirectional_gradient",
    target_layers=["model.layers.16.self_attn.o_proj"],
    projection_dim=256,
    learning_rate=1e-5
)

orchestrator = SynapseOrchestrator(
    nodes=[model_a, model_b],
    config=config
)

# Execute synchronized inference pass with live weight co-adaptation
with orchestrator.active_synapse():
    output_a, output_b = orchestrator.step(
        input_ids_a=tokenized_prompt_a,
        input_ids_b=tokenized_prompt_b
    )

```
---

### Computational Mechanics & Mathematical Formulation

To achieve a true "neurological" effect on model weights without explicit weight-space merging (which fails due to permutation symmetries and parameter divergence), `neuro-sync` operates on three coupled mathematical layers during the forward-backward pass.

#### 1. Cross-Model Activation Projection

Let Model $A$ and Model $B$ possess hidden state tensors at transformer layer $l$ defined as $h_A^{(l)} \in \mathbb{R}^{B \times S \times D_A}$ and $h_B^{(l)} \in \mathbb{R}^{B \times S \times D_B}$ respectively, where $B$ is batch size, $S$ is sequence length, and $D$ is hidden dimensionality.

Because dimensions and latent spaces differ between independent architectures, we define a parameterized orthogonal projection matrix $P_{A \to B} \in \mathbb{R}^{D_A \times D_B}$ (constrained such that $P^T P = I$) to map Model $A$'s manifold into Model $B$'s reference frame:


$$\tilde{h}_A^{(l)} = h_A^{(l)} P_{A \to B}$$

During the forward pass, this projected activation is injected directly into Model $B$'s attention block via residual gating:


$$h_B^{(l)} \leftarrow (1 - \alpha) h_B^{(l)} + \alpha \tilde{h}_A^{(l)}$$

#### 2. Gradient Cross-Feeding & Loss Landscape Coupling

The core of the "mind meld" occurs during backpropagation. Standard backprop computes gradients local to each model's independent loss $L_A$ and $L_B$. `neuro-sync` intercepts the backward pass to cross-feed gradients across the inter-process communication (IPC) boundary:

$$\frac{\partial L_{\text{total}}}{\partial h_B^{(l)}} = \frac{\partial L_B}{\partial h_B^{(l)}} + \lambda \cdot P_{A \to B} \left( \frac{\partial L_A}{\partial h_A^{(l)}} \right)$$

By injecting $\frac{\partial L_A}{\partial h_A^{(l)}}$ directly into Model $B$'s backward accumulation graph, Model $B$'s optimizer updates its parameters based on a joint gradient manifold. This forces both models' optimization trajectories to synchronize in real time.

#### 3. Hebbian Plasticity & Homeostatic Clamping

Unconstrained gradient cross-feeding across heterogeneous models inevitably causes gradient explosion or catastrophic forgetting. To enforce stability, parameter updates $\Delta W$ applied via shared memory-mapped LoRA adapters ($W = W_0 + \Delta W$, where $\Delta W = \frac{\alpha}{r} A B^T$) are governed by a modified Hebbian plasticity rule with homeostatic clamping:

$$\Delta W_{t+1} = \eta \left( \nabla_W L_{\text{total}} \cdot \mathcal{H}(\sigma) \right) - \gamma W_t$$

Where $\mathcal{H}(\sigma)$ is a homeostatic gating function that dampens updates when the cosine similarity between the models' internal hidden states falls below a critical threshold $\tau$:


$$\mathcal{H}(\sigma) = \frac{2}{1 + e^{-k(\text{sim}(h_A, h_B) - \tau)}} - 1$$
