# Kimi K3: The First Open-Weight Model at the 3-Trillion-Parameter Scale — A Deep Technical Dive

> **Author:** Imrankhan<br>
> **Date:** July 30, 2026<br>
> **Paper:** [Kimi K3 Technical Report (arXiv 2607.24653)](https://arxiv.org/abs/2607.24653)<br>
> **Weights:** [moonshotai/Kimi-K3 on Hugging Face](https://huggingface.co/moonshotai/Kimi-K3)<br>

---
<details>
<summary><b>Table of Contents</b></summary>

1. [Executive Summary](#1-executive-summary)
2. [The Big Picture: Three Axes of Information Flow](#2-the-big-picture-three-axes-of-information-flow)
3. [Full Architecture](#3-full-architecture)
4. [Architecture at a Glance — The Full K2 → K3 Diff](#4-architecture-at-a-glance-the-full-k2-k3-diff)
5. [Deep Dive: Kimi Delta Attention (KDA) — Sequence Mixing](#5-deep-dive-kimi-delta-attention-kda-sequence-mixing)
6. [Deep Dive: Attention Residuals (AttnRes) — Depth Mixing](#6-deep-dive-attention-residuals-attnres-depth-mixing)
7. [Deep Dive: Stable LatentMoE — Width Mixing](#7-deep-dive-stable-latentmoe-width-mixing)
8. [Quantile Balancing — Fixing Dying Experts at Scale](#8-quantile-balancing-fixing-dying-experts-at-scale)
9. [Stability Engineering at 2.78T Scale](#9-stability-engineering-at-278t-scale)
10. [No Positional Encoding — NoPE at Frontier Scale](#10-no-positional-encoding-nope-at-frontier-scale)
11. [Native Multimodal Vision: MoonViT-V2](#11-native-multimodal-vision-moonvit-v2)
12. [Pre-Training: Data, Scaling, and the 1M-Token Context](#12-pre-training-data-scaling-and-the-1m-token-context)
13. [Post-Training: Nine Specialists, Then a Merge](#13-post-training-nine-specialists-then-a-merge)
14. [The RL Environment Stack: 51 Million Sandboxes](#14-the-rl-environment-stack-51-million-sandboxes)
15. [Infrastructure: Making a 2.78T Model Actually Run](#15-infrastructure-making-a-278t-model-actually-run)
16. [Benchmark Results & Performance Analysis](#16-benchmark-results-performance-analysis)
17. [The Kimi Lineage: From K1.5 to K3](#17-the-kimi-lineage-from-k15-to-k3)
18. [Practical Implications: Serving & Inference](#18-practical-implications-serving-inference)
19. [Conclusion & What This Means for the Field](#19-conclusion-what-this-means-for-the-field)
20. [References](#20-references)

</details>

---

## 1. Executive Summary

On **July 27, 2026**, Moonshot AI released **Kimi K3** — a **2.78-trillion-parameter** open-weight Mixture-of-Experts (MoE) model with **104.2 billion active parameters per token**, a native **1-million-token context window**, and built-in vision capabilities. It is the **first open-weight model in the 3T-parameter class**.

<div align="center">
    <img src="https://sebastianraschka.com/images/blog/2026/kimi-k3-architecture-notes/kimi-k3-architecture.webp" alt="Kimi K3 Complete Architecture Diagram" width="680"/>
    <p><em>Figure 1: Complete Kimi K3 architecture with KDA, Gated MLA, Attention Residuals, LatentMoE, vision pathway, and benchmark comparisons. Credit: Sebastian Raschka.</em></p>
</div>

### Key Numbers at a Glance

| Metric | Value |
|---|---|
| **Total Parameters** | 2.78 Trillion |
| **Active Parameters / Token** | 104.2 Billion |
| **Total Experts** | 896 routed + 2 shared |
| **Active Experts / Token** | 16 |
| **Sparsity Ratio** | 56:1 (experts), 3.7% (parameters) |
| **Context Window** | 1,000,000 tokens |
| **Layers** | 93 (69 KDA + 24 Gated MLA) |
| **Hidden Dimension** | 7,168 |
| **Attention Heads** | 96 |
| **KV Compressed Dimension** | 512 |
| **Latent Expert Width** | 3,584 (½ of hidden dim) |
| **Scaling Efficiency vs K2** | ~2.5× |
| **Precision** | MXFP4 weights, MXFP8 activations |
| **Vocabulary** | 102,400 |
| **MTP (Multi-Token Prediction) Layers** | 1 |
| **AttnRes Block Size** | 12 (8 blocks + 1 short) |



The real surprise isn't the parameter count — it's that three-quarters of the attention layers use **linear attention** (not softmax), there is **no positional encoding anywhere**, and the residual stream has been replaced with **attention over depth**. K3 makes all three bets at once and reports a **~2.5× improvement in scaling efficiency**.

---


## 2. The Big Picture: Three Axes of Information Flow

Most model releases in 2025–26 scaled one thing. Kimi K3 is unusual because it scales a foundation — and it does so by attacking a single question from three angles simultaneously:

<div align="center">
    <img src="posts/images/kimi_1.png" alt="Three Axes of Information Flow in Kimi K3" width="700">
</div>


A transformer moves information in three directions — **along the sequence, up through depth**, and across **width**. Each has a default mechanism that’s been essentially frozen since 2017: softmax attention, the residual stream, and a dense feed-forward. **K3** replaces all three at once — and the payoff compounds into roughly a **2.5× improvement in scaling efficiency**.




| Axis | Default (2017) | K3 Replacement | What It Buys |
|---|---|---|---|
| **Sequence** | Softmax Attention | **Kimi Delta Attention** | Constant-memory token mixing at 1M context |
| **Depth** | Residual Stream (`h_l = h_{l-1} + f(h_{l-1})`) | **Attention Residuals** | Selective cross-layer information retrieval |
| **Width** | Dense FFN | **Stable LatentMoE** | Half-width expert routing for 896 experts |


 K3 is not primarily a bigger-model story — it’s an efficiency story that happens to end in a bigger model. Every headline change is a way to buy more capability per unit of compute, memory, or interconnect bandwidth. That’s why the paper spends more pages on kernels and sandboxes than on architecture.



---
## 3. Full Architecture

<div align="center">
    <img src="posts/images/k3_full.png" alt="Full Kimi K3 Hybrid Architecture Diagram" width="700">
</div>

**The whole architecture.** One block = 3 × (KDA + Stable LatentMoE) then 1 × (Gated MLA + Stable LatentMoE).

### A New Attention Architecture (Hybrid KDA + Gated MLA) :

The biggest architectural innovation is replacing traditional Transformer attention with a hybrid approach. Instead of using full attention in every layer, Kimi K3 repeats a pattern of:

- 3 Kimi Delta Attention (KDA) layers
- 1 Gated MLA layer

so around 75% of the model uses lightweight linear attention, while 25% still uses full global attention whenever exact information retrieval is needed.

The result is a model that can process extremely long documents without the massive computational cost of traditional Transformers.


---

## 4. Architecture at a Glance — The Full K2 → K3 Diff

Almost nothing survived the K2 → K3 transition untouched. Here is the full architectural comparison:



| Kimi K2 | Kimi K3 |
|---------|---------|
| <img src="https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/kimi-k2-1-trillion.webp" width="400"/> | <img src="https://sebastianraschka.com/llm-architecture-gallery/images/architectures/kimi-k3.webp" width="400"/> |
| *Figure 1: Kimi K2* | *Figure 2: Kimi K3* |





| Component | Kimi K2 (Jul 2025) | Kimi K3 (Jul 2026) | Change |
|---|---|---|---|
| **Total Parameters** | 1.04T | **2.78T** | 2.67× |
| **Active Parameters** | 32B | **104.2B** | 3.26× |
| **Hidden Dimension** | 7,168 | 7,168 | unchanged |
| **Layers** | 61 | **93** | +52% |
| **Attention Type** | MLA (all layers) | **3:1 KDA / Gated MLA hybrid** | new |
| **Positional Encoding** | RoPE | **NoPE** (none) | removed |
| **Total Experts** | 384 | **896** | 2.33× |
| **Active Experts** | 8 | **16** | 2× |
| **Sparsity Ratio** | 48 | **56** | ↑ (sparser) |
| **Expert FFN Type** | SwiGLU | **SiTU-GLU** | new |
| **MoE Architecture** | Standard MoE | **LatentMoE** (½-width) | new |
| **Residual Connection** | Standard add | **Attention Residuals** | new |
| **Routing Balance** | Bias nudge (DeepSeek-V3 style) | **Quantile Balancing** | new |
| **Optimiser** | MuonClip | **Per-Head Muon** | refined |
| **Vision** | Adapter-based (SigLIP init) | **Native MoonViT-V2** (from scratch) | new |
| **Precision** | BF16 | **MXFP4 / MXFP8** | lower |
| **Context Window** | 128K (extendable) | **1M** (native) | 7.8× |
| **Vocabulary** | 102,400 | 102,400 | unchanged |


---

## 5. Deep Dive: Kimi Delta Attention (KDA) — Sequence Mixing

<div align="center">
    <img src="posts/images/kad.png" alt="Kimi Delta Attention Recurrence Mechanism" width="600">
</div>

### 5.1 The Problem: Softmax Attention Doesn't Scale to 1M Tokens

Standard softmax attention keeps **every past token** as a key-value pair in memory. The KV cache grows linearly with context length — at 1 million tokens, this is computationally expensive . This is the fundamental bottleneck for long-context models.

### 5.2 The KDA Solution: Fixed-Size Recurrent Memory

KDA replaces softmax attention with a **linear-attention recurrence** that maintains a **fixed-size matrix state** $S_t$. This state is updated once per token with two key mechanisms:

**The core recurrence:**

$$S_t = \left(I - \beta_t \cdot k_t k_t^\top\right) \cdot \text{Diag}(\alpha_t) \cdot S_{t-1} + \beta_t \cdot k_t v_t^\top$$

$$\tilde{o}_t = S_t^\top \cdot q_t$$

Reading this equation right to left:

1. **$\text{Diag}(\alpha_t) \cdot S_{t-1}$** — **Channel-wise decay**: Decays the old state **per feature channel**. $\alpha_t \in (0,1)^{d_k}$ is the per-channel retention. Some dimensions persist for a million tokens while others flush every few hundred.

2. **$(I - \beta_t \cdot k_t k_t^\top)$** — **The delta rule**: Projects out whatever the current key already points at. It's an *update*, not an *append*. This erases exactly the slot about to be overwritten.

3. **$\beta_t \cdot k_t v_t^\top$** — **Write operation**: Writes the new key-value pair, with $\beta_t$ controlling write strength.

4. **$S_t^\top \cdot q_t$** — **Read operation**: Simple read of the state through the query.

### 5.3 Projections: How Q, K, V Are Computed

The query, key, and value projections use a **short convolution** for local context and **L2-normalisation** for stability:

$$q_t^h, k_t^h = \text{L2Norm}\left(\text{Swish}\left(\text{ShortConv}\left(W_{q/k}^h \cdot x_t\right)\right)\right)$$

$$\beta_t^h = \text{Sigmoid}\left(W_\beta^h \cdot x_t\right)$$

The **ShortConv** (kernel size 4, per the released config) lets each position peek at its immediate neighbours before the recurrence sees it — cheap local context that the linear state doesn't have to store. **L2-normalising q and k** bounds the delta-rule projection, critical for stability.

### 5.4 The $g_{\min}$ Bound: Algorithm–System Co-design

In the chunkwise-parallel implementation, recurrences are processed in chunks (GPUs hate serial work). This requires rescaling keys by the reciprocal of cumulative decay — which **can blow up without bound** and overflow in low precision.

**Kimi Linear's approach** used $g = -e^A \cdot \text{Softplus}(z)$, which runs to $-\infty$. This forced complex diagonal-tile handling.

**K3's fix** — an elegant bounded sigmoid:

$$g_t^h = g_{\min} \cdot \text{Sigmoid}\left(e^{A_h} \cdot z_t^h\right) \in (g_{\min}, 0)$$

$$\alpha_t^h = \exp(g_t^h)$$

With **$g_{\min} = -5$** (fixed) and $A_h$ learnable per head (initialised to 0):
- Every retention factor is at least $e^{-5} \approx 0.0067$
- Cumulative log-decay over a 16-token tile stays inside $(-80, 0)$
- The reciprocal stays under $e^{80}$ — comfortably inside BF16 range

> **The payoff is huge:** Because rescaling can no longer overflow, **every causal tile — diagonal included — becomes a dense Tensor Core matmul.** The special-cased position-pair path disappears entirely. A numerical guarantee bought a kernel simplification, which bought throughput. This is the paper's tightest example of algorithm–system co-design.

<div align="center">
    <img src="posts/images/k3_bound.png" alt="g_min Bounded Sigmoidal Log-Decay" width="700">
</div>

### 5.5 Chunkwise Parallel Form

The output for each chunk splits into inter-chunk (state from previous chunks) and intra-chunk (local interactions) terms:

$$O_{[t]} = \underbrace{\left(\Gamma_{[t]}^{1 \to C} \odot Q_{[t]}\right) S_{[t]}}_{\text{inter-chunk}} + \underbrace{A_{[t]} \tilde{V}_{[t]}}_{\text{intra-chunk}}$$

Only the first term is serial; the second is a dense matmul — making it GPU-friendly.

### 5.6 The 3:1 Hybrid Pattern

K3 alternates strictly at the layer level: **3 KDA layers → 1 Gated MLA (softmax) layer**. Across 93 layers:

```python
# K3's layer pattern (93 layers total)
layers = []
for block in range(23):  # 23 blocks of 4
    layers.extend(["KDA", "KDA", "KDA", "Gated_MLA"])
layers.append("Gated_MLA")  # Final extra MLA layer

# Result: 69 KDA + 24 Gated MLA = 93 layers
assert sum(1 for l in layers if l == "KDA") == 69
assert sum(1 for l in layers if l == "Gated_MLA") == 24
```

An extra Gated MLA layer is bolted onto the end — the very last thing before producing a token is always a **full global attention pass**. The released config confirms this with a `full_attn_layers` list containing exactly 24 entries.

Prototype results from Kimi Linear (the 48B precursor) showed:
- Up to **75% KV cache reduction**
- **6.3× decoding throughput** at 1M context
- Comparable quality across all benchmarks

>  **The whiteboard analogy:** Regular attention is a filing cabinet — every note you've ever written is still there, but finding things gets slower as it fills. Linear attention is a whiteboard: always the same size, always fast to read, but you have to *erase to write*. The delta rule is the discipline of **wiping the specific spot you're about to write on** rather than smearing new ink over old. The channel-wise forget gate lets different regions of the board fade at different speeds — the project deadline stays legible for months, the passing thought fades in minutes.

---

## 6. Deep Dive: Attention Residuals (AttnRes) — Depth Mixing

<div align="center">
    <img src="posts/images/att_res.png" alt="Attention Residuals Cross-Layer Information Retrieval" width="700">
</div>

### 6.1 The Residual Stream Bottleneck

Every transformer since 2017 has moved information through depth the same way:

$$h_l = h_{l-1} + f_l(h_{l-1})$$

A **single running sum.** The paper's observation is razor-sharp: *this is exactly the bottleneck that recurrence had over time, and which attention was invented to fix.* By the time you're deep in the network, early layers have been diluted to almost nothing, and layer 80 has **no way to say "I specifically want what layer 14 computed."**


### 6.2 How AttnRes Works

AttnRes applies the Transformer's own medicine to depth. Each layer $l$ carries a **learnable pseudo-query** $w_l$ — a parameter vector not derived from the input. Keys and values are the outputs of all preceding layers plus the token embedding:

$$\alpha_{i \to l} = \frac{\phi(q_l, k_i)}{\sum_{j=0}^{l-1} \phi(q_l, k_j)}$$

$$h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i$$

where the attention kernel uses **RMSNorm** — which is load-bearing:

$$\phi(q, k) = \exp\left(q^\top \text{RMSNorm}(k)\right)$$

Without RMSNorm, layers that produce large-magnitude outputs would dominate attention weights **purely by scale** rather than by relevance.

### 6.3 Block AttnRes: The Version That Actually Ships

Kimi K3’s innovation is that AttnRes no longer forces $O(L \cdot d)$ memory scaling. Instead, it leverages Kimi Delta Attention and Block-level summaries to keep depth attention lightweight, enabling practical 1M-token contexts.


K3 partitions its 93 layers into **8 blocks of 12** (`attn_res_block_size: 12`):

```python
# Block AttnRes partitioning
block_size = 12
n_layers = 93
n_blocks = math.ceil(n_layers / block_size)  # 8 blocks (last one is short)

# Inside a block: layer outputs are SUMMED into a running partial
# Across blocks: full attention over BLOCK-LEVEL SUMMARIES only
# Memory: O(Nd) with N ≈ 8, not O(Ld) with L = 93
```

The value matrix construction:

$$V = \begin{cases} [b_0, \ldots, b_{n-1}]^\top & \text{if } i = 1 \text{ (first layer of block)} \\ [b_0, \ldots, b_{n-1}, b_n^{i-1}]^\top & \text{if } i \geq 2 \end{cases}$$

The first layer of each block attends only over previous **block summaries**. Subsequent layers additionally see the running partial sum from within their own block. This lets the inter-block result be merged with the intra-block partial via **online softmax** — the same trick FlashAttention uses.


---

## 7. Deep Dive: Stable LatentMoE — Width Mixing
896 experts, 16 active. That’s a sparsity ratio of 56 — up from 48 in K2 — and it is where most of K3’s 2.78T parameters live. It’s also where things break.

<div align="center">
    <img src="posts/images/k3_moe.png" alt="Stable LatentMoE Width Mixing & Routing Diagram" width="700">
</div>

### 7.1 The Communication Bottleneck

A 2.78T-parameter model doesn't sit on one GPU — its experts are spread across **dozens of accelerators**. vLLM's day-zero guidance for K3 is an **8×B300 node at minimum**; realistic serving uses a 64-GPU supernode or NVL72 rack.

Every token **physically travels between GPUs** to reach the 16 experts chosen for it. In a conventional MoE, doubling the experts per token **doubles the interconnect traffic**.

### 7.2 The LatentMoE Trick: Separate Model Width from Expert Width

LatentMoE separates two things that are usually conflated: the model's width (7,168) and the width the routed experts work in (3,584 — exactly **half**):

$$u = \sum_{i \in T_k(x)} p_i \cdot E_i^{\text{routed}}\left(W_\downarrow \cdot x\right)$$

$$y = \sum_{j=1}^{N_s} E_j^{\text{shared}}(x) + W_\uparrow \cdot \text{RMSNorm}(u)$$

**Two paths, summed:**
1. **Shared path** ($N_s = 2$ experts, always fire): Full-width $d = 7168$, for transformations everything needs.
2. **Routed path** (16 of 896 experts): Down-project → dispatch → aggregate → RMSNorm → up-project.

The RMSNorm between aggregation and up-projection is a **K3 addition** — the original LatentMoE applied $W_\uparrow$ directly to $u$, whose scale swung wildly with which experts happened to fire.

```python
# Simplified LatentMoE forward pass
def latent_moe_forward(x, router, shared_experts, routed_experts, W_down, W_up):
    """x: [batch, seq_len, d_model=7168]"""
    
    # Shared path — always on, full width
    shared_out = sum(expert(x) for expert in shared_experts)  # 2 experts
    
    # Routed path — half width
    x_latent = W_down @ x                # 7168 → 3584 (half width!)
    scores = router(x)                    # [batch, seq, 896]
    top_k_indices = scores.topk(16)       # Select 16 of 896
    
    # Dispatch to selected experts in latent space
    expert_outputs = []
    for idx in top_k_indices:
        expert_outputs.append(scores[idx] * routed_experts[idx](x_latent))
    routed_out = sum(expert_outputs)
    
    # Normalise and project back up
    routed_out = rms_norm(routed_out)     # Stabilise scale
    routed_out = W_up @ routed_out        # 3584 → 7168 (back to full width)
    
    return shared_out + routed_out
```

LatentMoE’s width mixing halves the communication payload, enabling K3 to scale across 64–72 GPU clusters with ~2.5× efficiency gains. The RMSNorm stabilizes routed outputs, preventing gradient spikes, while auxiliary load‑balancing ensures experts are evenly utilized. Inference still requires cross‑GPU routing, but the reduced payload makes long‑context serving practical.

---

## 8. Quantile Balancing — Fixing Dying Experts at Scale

### 8.1 The Problem at ~10³ Experts

With 896 experts per layer, load balancing stops being a nuisance and becomes a **correctness problem**. Left alone:
- A few experts get **everything**
- Some never fire and quietly stay **untrained** (dying experts)
- The whole layer runs at the speed of the **busiest GPU**

### 8.2 The Old Fix: DeepSeek-V3's Bias Nudge

K3 uses auxiliary-loss-free routing: an expert-specific bias $b_j$ is added to the router score *for selection only* and left out of mixture weights so it never distorts the gradient:

$$T_i = \text{argtopk}(s_i + b)$$

$$p_{i,j} = \frac{s_{i,j}}{\sum_{r \in T_i} s_{i,r}}$$

The old DeepSeek-V3 update nudges this bias by a fixed step: $b_j^{(t+1)} = b_j^{(t)} + \gamma \cdot \text{sign}(\ldots)$. That $\gamma$ is a nasty hyperparameter — too small and it never catches up, too large and loads oscillate.

### 8.3 Quantile Balancing: One Exact Jump, No Learning Rate

<div align="center">
    <img src="posts/images/qb.png" alt="Quantile Balancing vs Sign-SGD Load Balancing" width="700">
</div>

**Quantile Balancing** throws out the step size entirely:

$$\hat{b}_j^{(t+1)} \leftarrow -\text{quantile}_{1-k/n}\left(s_{:,j} - \alpha^{(t)}\right)$$

$$b^{(t+1)} \leftarrow \hat{b}^{(t+1)} - \text{mean}\left(\hat{b}^{(t+1)}\right) \cdot \mathbf{1}$$

**How it works:**
1. Routing runs **Top-(k+1)** instead of Top-k. The (k+1)-th entry gives the **cutoff** $\alpha_i$ — the score an expert must beat to make a token's shortlist.
2. The margin $s_{i,j} - \alpha_i$ says how comfortably expert $j$ would have made it.
3. Setting the bias to the $(1-k/n)$-quantile of those margins means **exactly the target load** of tokens stay above the line.
4. Mean-centre keeps biases from drifting.

The appendix derives QB from the **dual of a maximum-score balanced assignment LP** — the same problem BASE Layers solved with explicit auction algorithms. The old sign-based rule turns out to be plain SignSGD on that dual objective. QB jumps directly to the exact coordinate minimiser. **That's why it needs no step size.**

> **The hospital analogy:** You run a hospital with 896 specialists and each patient sees 16. The old fix: "that doctor looks busy, make them *slightly* less attractive tomorrow." Nudge too gently and the queue never clears; too hard and everyone stampedes. Quantile Balancing asks: **"if each doctor should see exactly 2,000 patients, how good would a referral have to be to make their list?"** — then sets the bar exactly there, in one move.

---

## 9. Stability Engineering at 2.78T Scale

### 9.1 SiTU-GLU: Replacing SwiGLU with Bounded Activations

SwiGLU multiplies two **unbounded** quantities — if two large values line up, the outlier in MXFP4 precision becomes an **overflow**. SiTU-GLU (Sigmoid Tanh Unit GLU) smoothly caps both branches:

$$\text{SiTU-GLU}(x) = \left[\beta_1 \tanh\left(\frac{W_g x}{\beta_1}\right) \odot \text{Sigmoid}(W_g x)\right] \odot \left[\beta_2 \tanh\left(\frac{W_u x}{\beta_2}\right)\right]$$

With $\beta_1 = 4$ and $\beta_2 = 25$ (confirmed in the released config):

$$\|\text{SiTU-GLU}(x)\|_\infty \leq \beta_1 \cdot \beta_2 = 100$$

**Why it works near zero:** Because $\beta \tanh(z/\beta) = z + O(z^3/\beta^2)$, SiTU-GLU matches SwiGLU **to first order** near the origin. Identical behaviour where activations normally live; diverges only in the tail where SwiGLU would have exploded.

```python
import torch
import torch.nn.functional as F

def swiglu(x, W_gate, W_up):
    """Standard SwiGLU — UNBOUNDED output"""
    return F.silu(x @ W_gate.T) * (x @ W_up.T)

def situ_glu(x, W_gate, W_up, beta1=4.0, beta2=25.0):
    """SiTU-GLU — output bounded by beta1 * beta2 = 100"""
    gate = x @ W_gate.T
    up = x @ W_up.T
    
    # Bounded gate branch
    gate_bounded = beta1 * torch.tanh(gate / beta1) * torch.sigmoid(gate)
    # Bounded up branch
    up_bounded = beta2 * torch.tanh(up / beta2)
    
    return gate_bounded * up_bounded
    # Max possible output: 4 * 25 = 100 (vs. unbounded for SwiGLU!)
```

### 9.2 Per-Head Muon Optimiser

Muon's idea: "square up" a matrix-shaped update so no single direction dominates. But Q/K/V projections aren't one matrix — they're **96 heads stacked together**. Orthogonalising the stack lets loud heads set the update scale for quiet ones.

**Per-Head Muon:** Partition the momentum matrix along the head dimension and orthogonalise each head's block separately. This **equalises learning dynamics** across heads and is actually *cheaper* — Newton–Schulz on thin per-head blocks costs less than on the full projection.

| Step | Paper | Key Change |
|---|---|---|
| **Muon** | Jordan et al. | Matrix-aware Newton–Schulz orthogonalisation |
| **Moonlight** | Moonshot AI | Scaled to 3B–16B range |
| **MuonClip (K2)** | K2 report | + QK-Clip: rescale heads if max logit > τ=100. Zero loss spikes over 15.5T tokens |
| **Per-Head Muon (K3)** | K3 report | Orthogonalise per head, not per projection. Cheaper + better equalised |

### 9.3 MXFP4 Quantisation-Aware Training

- MoE expert weights → **MXFP4** (4-bit)
- Activations → **MXFP8** (8-bit)
- Everything else (attention projections, latent MoE projections, shared experts, routers) → higher precision

Crucially, QAT runs through the **entire** post-training stage including RL, and **rollout and training share the same quantisation scheme** — eliminating the train–inference mismatch that normally makes quantised RL fragile. The released weights *are* the trained precision.

---

## 10. No Positional Encoding — NoPE at Frontier Scale

One of the most under-discussed choices in K3: **no RoPE, no ALiBi, nothing, anywhere.** Position is carried entirely by KDA's recurrence and decay.

[Sebastian Raschka flagged this](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html) as particularly noteworthy:

> *"Interestingly, Kimi K3 got rid of all RoPE layers and uses NoPE (No Positional Embeddings) everywhere instead... There were a few architectures that only used NoPE everywhere, but this is the first frontier-level one as far as I know."*

**The payoff is transformative:** Because there is no positional parameterisation:
- Extending to 1M tokens requires **no RoPE base retuning**
- **No YaRN**, no interpolation
- The model just **extrapolates**

Positional encoding is usually one of the most fiddly parts of long-context work. K3 **deletes the problem** rather than solving it — the released config confirms: `mla_use_nope: true`.

---

## 11. Native Multimodal Vision: MoonViT-V2

### 11.1 Training From Scratch (Not SigLIP)

Standard practice: bolt on a contrastively pre-trained vision encoder (SigLIP). K3 trains its **401M-parameter, 27-layer encoder entirely from scratch** with next-token prediction, jointly with the language model from step zero.

**The reason isn't quality — it's stability.** A SigLIP-initialised tower shows persistently higher gradient norms with frequent spikes during joint optimisation; the from-scratch tower stays flat. And it **matches SigLIP on every vision benchmark**.

<div align="center">
    <img src="posts/images/k3_v_train.png" alt="MoonViT-V2 Vision Encoder Training From Scratch" width="700">
</div>

### 11.2 Vision Architecture Details

The vision path includes careful engineering:
- **RMSNorm** throughout, all bias terms removed (stabilises from-scratch training)
- Attention factorised into **intra-frame spatial** and **inter-frame temporal** passes
- Fully shared parameters between images and video
- **2×2 pixel-shuffle** before projection — cuts visual tokens **4×**, making 3584×3584-pixel inputs affordable inside a 1M-token budget


---

## 12. Pre-Training: Data, Scaling, and the 1M-Token Context

### 12.1 The 2.5× Scaling Efficiency Claim

The headline efficiency number is a **fitted scaling-law curve**, not a single measurement. Moonshot re-ran dedicated scaling-law studies and fitted loss-versus-FLOPs curves for K2 and K3 on held-out out-of-distribution validation data.

<div align="center">
    <img src="posts/images/k3_tranning.png" alt="Kimi K3 Pre-Training Loss & Scaling Efficiency Curves" width="700">
</div>

**What 2.5× means:** To reach a given validation loss, K3's architecture needs about **1/2.5 of the compute K2's did** — same loss for ~40% of the FLOPs.

> **What it doesn't mean:** The paper does not decompose it — we're never told how much comes from KDA versus AttnRes versus LatentMoE versus "refined data and training recipes." It also isn't a claim about downstream capability per dollar.

**Methodology note worth borrowing:** The team found their scaling study **consistently favoured cosine decay over Warmup-Stable-Decay** — but only once they ran an independent hyperparameter search for each schedule. Under each schedule's own optimum, cosine won.

### 12.2 Data Pipeline

Four text domains plus a large vision corpus:

| Domain | Processing |
|---|---|
| **Web Text** | Rule-based filtering, classifier quality scoring, deduplication |
| **Code** | Multi-language, with rendered visual outputs |
| **Mathematics** | Rephrasing with style/perspective diversity (from K2) |
| **Knowledge** | Rephrasing with fidelity verification |
| **Vision** | Captions, interleaved image-text, OCR, perception, video, visual coding |

**Key data innovations:**
- **Rephrasing** from K2: Ten diverse rephrasings for one epoch beat repeating raw data (SimpleQA: 23.76% → 28.94%)
- **Coordinate supervision** in both absolute pixels and normalised [0,1] for resolution-robust localisation
- **Programmatic multimodal data**: Code snippets paired with their rendered visuals — SVG, 3D assets, webpages, games, CAD schematics

>  **Not disclosed:** The total pre-training token count appears **nowhere** — not in the report, not the blog, not the model card. Neither does total training compute, GPU-hours, cluster size, or dollar cost.

### 12.3 Getting to a Million Tokens

The context curriculum has four stages, deliberately back-loaded:

$$\text{8K} \xrightarrow{\text{pre-training}} \text{64K} \xrightarrow{\text{cooldown}} \text{256K} \xrightarrow{\text{cooldown}} \text{1M}$$

Two things make this work:

1. **NoPE means there's nothing to extend.** The model extrapolates to 1M directly.

2. **Length alone doesn't buy long-range ability.** Naturally-occurring long documents are full of near-duplicates and machine-generated logs. So the team **synthesises** long-context data by permuting and concatenating multimodal documents — **so that the embedded task is only solvable by attending to information scattered across the full window.** Without that, attention degenerates into local patterns and you get a model with a 1M context that only really uses the last 4K.



---

## 13. Post-Training: Nine Specialists, Then a Merge

This is the biggest structural change from K2's post-training. The pipeline has **three stages**:

<div align="center">
    <img src="posts/images/k3_post.png" alt="Three-Stage Post-Training Pipeline & MOPD Distillation" width="700">
</div>

### 13.1 Stage 1: SFT Cold Start

Trajectories synthesised using domain-specialised models, put through multi-stage verification and human-in-the-loop annotation. Everything serialised with **XTML** (eXtensible Token Markup Language):
- Angle brackets replaced by reserved special tokens (`[open]`, `[sep]`, `[close]`, `[end_of_msg]`)
- Eliminates tokenisation ambiguity
- Enables grammar-constrained decoding

The assistant message body splits into three channels — `think`, `response`, and `tools` — credited to OpenAI's Harmony format.

### 13.2 Stage 2: RL Across 3 Domains × 3 Efforts

Nine separate RL experts trained independently:

| | **Low Effort** | **High Effort** | **Max Effort** |
|---|---|---|---|
| **General** | Quick answers, search | Multi-step reasoning | Deep research, exhaustive |
| **General Agents** | Simple assistant tasks | Long-horizon workflows | Multi-day simulated tasks |
| **Coding Agents** | Code fixes, snippets | Repo-level engineering | GPU kernels, full-stack apps |

**Reasoning-effort control** uses a reward penalty, not a token cap:

$$r(y | x) = \begin{cases} -1 & \text{if } T(y) > \tau \cdot b_0(x) \\ r_{\text{task}}(y | x) & \text{otherwise} \end{cases}$$

Each problem $x$ gets an initial token budget $b_0(x)$ from the cold-start model. $\tau$ is annealed down across a curriculum to produce low- and high-effort experts.

### 13.3 Stage 3: Multi-Teacher On-Policy Distillation (MOPD)

Nine experts are not shippable. MOPD folds them into one model using a **dense per-token reward**:

$$r_{\text{opd}}^d(y_t | e, x, y_{<t}) = \text{clip}\left(\text{sg}\left[\log \frac{\pi_{\text{teacher}}^{(d,e)}(y_t | x, y_{<t})}{\pi_\theta(y_t | e, x, y_{<t})}\right], -R_{\max}, R_{\max}\right)$$

The reward on each token is the **log-ratio of teacher to student probability**, stop-gradiented and clipped. Big gap = strong positive signal. Because it's a dense per-token reward plugging into the existing RL framework, it inherits partial rollouts for free.


---

## 14. The RL Environment Stack: 51 Million Sandboxes

Architecture is publishable. **Environments are the moat.**

### 14.1 Randomised Agent Harness

The harness itself is a collection of configurable, composable modules. Different configurations can instantiate **Kimi Code, Claude Code, Codex, OpenClaw and Hermes** — plus harnesses that don't exist outside training. During RL, different configurations are sampled **per task group**, so the model never overfits to one convention.

### 14.2 Knowledge-Graph-Guided Task Synthesis

Agents build a **self-evolving hierarchical knowledge graph** — a DAG grown by recursive, agent-driven web exploration. Sampling nodes at varying granularity produces keyword sets → web queries → real materials → tasks. It's a machine for manufacturing specialised, non-duplicated training tasks in domains you'd never enumerate by hand.

<div align="center">
    <img src="posts/images/kn_graph.png" alt="Knowledge-Graph-Guided Task Synthesis Pipeline" width="700">
</div>

### 14.3 Five Environment Families

| Family | Description | Key Feature |
|---|---|---|
| **GPU Kernel Optimisation** | CUDA, Triton, CuTe, Gluon, ThunderKittens | Graded reward: 0 → 0.5 → 1.0 approaching roofline. Anti-hacking detection |
| **Personal Assistant** | Mock Gmail, Notion, Slack, Canvas | Multi-day persistent environments, thousands of tool calls |
| **Autonomous Execution** | Goal + tools + budget + verifier, no reference trajectory | Rewards from final environment state, not self-report |
| **Web Development** | Websites, games, 3D/WebGL, data viz, full-stack | Zeroed if project fails to build or fakes the artifact |
| **Verifiable Agentic Problems** | Multi-step search, professional workflows, visual reasoning | Investment banking, legal practice, STEM visual puzzles |

### 14.4 The Sandbox Infrastructure: AgentENV

- **51 million+** isolated sandbox environments executed during training
- **Fork/clone**: Running sandboxes can clone into independent child environments
- **Three-layer snapshot system**: builder staging → committed repository → node-local runtime
- **Persistent rollout & state** across long-horizon tasks

---

## 15. Infrastructure: Making a 2.78T Model Actually Run

### 15.1 KDA Context Parallelism (KCP)

Standard context parallelism splits a long sequence across GPUs. For softmax attention, ranks exchange KV blocks. For linear attention, the standard trick (sum local states) is **wrong for KDA** because the delta rule acts on the incoming state.

K3's solution: each rank computes two things locally:
- **Cumulative transition** $M^{t \leftarrow 1}$ (product of all local $M_t$ matrices)
- **Locally-generated state** $\tilde{S}$ (recurrence started from zero)

<div align="center">
    <img src="posts/images/kda_par.png" alt="KDA Context Parallelism Matrix Composition" width="700">
</div>

$$S_{[i+1]}^t = \tilde{S}_{[i+1]}^t + M_{[i+1]}^{t \leftarrow 1} \cdot S_{[i]}^{T_i}$$

These compose **associatively** — one fixed-size all-gather + prefix scan reconstructs every rank's exact incoming state, with linear compute scaling.

### 15.2 MoonEP: Perfect Expert-Parallel Load Balance

MoonEP achieves **perfect** balance using dynamic redundant experts, proven to need at most $E/R$ redundant slots per rank. This means:
- Planning always admits a solution (no training stops)
- Communication buffers shrink from $S \times K \times R$ to fixed $S \times K$
- Computation shapes become **statically known**
- Zero per-layer host–device synchronisation stalls

### 15.3 Memory Engineering

At 2.78T parameters, memory engineering is a catalogue of individually small ideas:
- **Unified activation manager**: Recomputation, quantisation, and offload as composable storage policies
- **Gradient dependency rewriting**: Eliminate stored tensors at the cost of cheap element-wise ops
- **Cross-GPU activation offloading**: Activations remotely offloaded to other pipeline ranks via Mooncake Transfer Engine
- **P2P Muon orthogonalisation**: Peer-to-peer parameter fetching instead of all-gather

---


## 16. Benchmark Results & Performance Analysis

K3 lands just behind Claude Fable 5 and GPT-5.6 Sol on most benchmarks and **consistently ahead of everything else** — proprietary or open.

### 16.1 Headline Results

| Benchmark | Kimi K3 | Claude Fable 5 | GPT-5.6 Sol | Notes |
|---|---|---|---|---|
| **BrowseComp** | **91.2** | 89.4 | 88.7 | #1 at \$2.03/task (½ GPT-5.6 cost) |
| **GPQA Diamond** | **93.5** | 92.1 | 91.8 | Highest for open-weight at launch |
| **Terminal-Bench 2.1** | 88.3 | 90.1 | 87.5 | |
| **FrontierSWE** | 81.2 | 83.7 | 80.9 | |
| **ProgramBench** | 77.8 | 79.2 | 78.1 | |
| **DeepSWE** | 67.5 | 69.4 | 66.8 | |
| **SWE Marathon** | 42.0 | 44.1 | 43.2 | |
| **Humanity's Last Exam** | 56.0 | **63.0** | 58.5 | K3's honest gap |
| **CritPt** | 23.4 | 28.6 | **32.3** | Hardest reasoning |

### 16.2 Independent Rankings

| Leaderboard | K3 Ranking | Score |
|---|---|---|
| **Artificial Analysis Intelligence Index** | **4th of 580** | 57.1 |
| **Vals Index** | **2nd of 39** | — |
| **WebDev Arena** | **#1 of 99** | First open model ever to top it |
| **Frontend Code Arena** | **#1** | Elo 1,679 (vs Fable 5: 1,631) |

### 16.3 Autonomous Execution: Camera Repair System

In the "black-box system replication" benchmark (reconstruct a hidden 3D-camera repair system as a web app using only oracle queries):

| Model | Completion |
|---|---|
| **Kimi K3** | **1.000** (full) |
| Claude Opus 4.8 | 0.918 |
| GPT-5.5 | 0.893 |
| Kimi K2.6 | 0.560 |

---

## 17. The Kimi Lineage: From K1.5 to K3

Almost none of K3's ideas originated with K3 — each is the current end of a chain:

```
Jan 2025  ─── Kimi k1.5
                 └── Partial rollouts for RL (still used in K3)

Jul 2025  ─── Kimi K2 (1.04T, 384 experts, top-8)
                 ├── MuonClip / QK-Clip → zero loss spikes over 15.5T tokens
                 └── Data rephrasing

Oct 2025  ─── Kimi Linear (48B)
                 ├── KDA (Gated DeltaNet → per-channel decay)
                 ├── 3:1 hybrid pattern (KDA / Gated MLA)
                 ├── NoPE everywhere
                 └── Attention Residuals (first version)

Feb 2026  ─── K2.5 (1.04T backbone + native vision)
Apr 2026  ─── K2.6 (1.04T backbone + agentic RL swarm)

Jul 2026  ─── Kimi K3 (2.78T, 896 experts, top-16)
                 ├── LatentMoE (from Nemotron 3 lineage)
                 ├── Quantile Balancing (from balanced-assignment LP dual)
                 ├── SiTU-GLU (bounded activation)
                 ├── Per-Head Muon (orthogonalise per head)
                 ├── MoonViT-V2 (vision from scratch)
                 ├── g_min bound (overflow-free KDA tiles)
                 ├── Block AttnRes (practical depth attention)
                 └── MOPD (9-to-1 teacher distillation)
```

### Borrowed from Other Labs

| Component | Origin Lab |
|---|---|
| **MLA** (Multi-head Latent Attention) | DeepSeek-V2 |
| **Shared + routed experts** | DeepSeekMoE |
| **Auxiliary-loss-free routing** | DeepSeek-V3 |
| **LatentMoE** (down-project → route → up-project) | Nemotron 3 (NVIDIA) |
| **Gated DeltaNet** (delta rule + decay recurrence) | Yang et al. |
| **Muon** optimiser | Keller Jordan |
| **EAGLE-3** speculative decoding | Li et al. |
| **Harmony** channel format | OpenAI |

> What K3 contributes is the **combination**, plus the specific fixes each idea needed to survive at 2.78T.

---

## 18. Practical Implications: Serving & Inference

### 18.1 Hardware Requirements & Serving Infrastructure

| Deployment | Configuration |
|---|---|
| **Minimum Node** | 8× NVIDIA B300 (or equivalent FP8/FP4 tensor core node) |
| **Production Serving** | 64-GPU supernode / NVL72 rack with high-speed NVLink interconnect |
| **Serving Framework** | vLLM (day-zero support for LatentMoE and KDA chunkwise execution) |

### 18.2 Speculative Decoding & Quantization Strategy

- **EAGLE-3 Speculative Draft Model:** Employs EAGLE-3 speculative decoding to accelerate token generation for long contexts without sacrificing output distribution quality.
- **Native MXFP4/MXFP8 Execution:** Since expert weights are natively trained in MXFP4 and activations in MXFP8, deployment requires zero post-hoc quantization loss during serving.
- **KV Cache Footprint:** Thanks to the 3:1 KDA / Gated-MLA hybrid design, KV cache memory footprint is reduced by up to 75% at 1M context, drastically lowering VRAM pressure per user session.

---

## 19. Conclusion & What This Means for the Field

Kimi K3 answers an open question: **Can non-standard architectures — linear attention, NoPE, latent MoE, attention over depth — scale to the 3T class without falling apart?**

The answer is yes, and the implications are significant:

1. **The 1T ceiling is broken.** Open models no longer need to cluster around the same parameter count.

2. **Efficiency compounds.** Seven independent improvements — each modest alone — compound into a 2.5× scaling efficiency gain. This is compound interest applied to model design.

3. **Foundation quality matters.** K3's performance validates the "study harder, not just think longer" thesis.

4. **NoPE works at frontier scale.** First evidence that positional encoding can be entirely eliminated from a frontier model.

5. **Native multimodality from scratch is viable.** Pre-trained vision encoders may be unnecessary.

6. **Environments are the moat.** K3's 51M sandbox infrastructure for agentic RL is as consequential as its architecture — and much harder to replicate.

K3 is not the best model on every benchmark. But it's the **strongest open-weight model ever released**, and it proves the frontier is not the exclusive province of closed labs. The first open model at the 3-trillion-parameter scale is here — built on ideas that anyone can read, implement, and improve.

I hope you had a great time reading this. Share an interesting idea or just drop in a Hi at laptop14072024@gmail.com! Until next time 👋

---

## 20. References

1. **Kimi K3 Technical Report.** Kimi Team, Moonshot AI. arXiv:2607.24653. [arxiv.org](https://arxiv.org/abs/2607.24653)

2. **Sebastian Raschka.** "Kimi K3 Architecture Notes." July 28, 2026. [sebastianraschka.com](https://sebastianraschka.com/blog/2026/kimi-k3-architecture-notes.html)

3. **Kimi K3 Visual Walkthrough.** [k3kimi.netlify.app](https://k3kimi.netlify.app/)

4. **Kimi K3 on Hugging Face.** [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)

5. **Sebastian Raschka.** "LLM Architecture Gallery." [sebastianraschka.com/llm-architecture-gallery](https://sebastianraschka.com/llm-architecture-gallery/)

6. **Yang et al.** "Gated DeltaNet." 2024. (delta rule + decay recurrence)

7. **DeepSeek-V2/V3 Technical Reports.** DeepSeek AI. (MLA, auxiliary-loss-free routing)

8. **Kimi k1.5 Technical Report.** Moonshot AI, January 2025. (partial rollouts)

9. **Kimi K2 Technical Report.** Moonshot AI, July 2025. (MuonClip, data rephrasing)

10. **Kimi Linear Technical Report.** Moonshot AI, October 2025. (KDA prototype at 48B)

11. **Nemotron 3 Ultra.** NVIDIA. (LatentMoE lineage)



---

