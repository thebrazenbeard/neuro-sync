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
