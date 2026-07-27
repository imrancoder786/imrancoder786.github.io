# Building the Transformer Architecture from Scratch in PyTorch: An In-Depth Mathematical & Implementation Guide

> **Author:** Imrankhan<br>
> **Project:** Sequence-to-Sequence English-to-Tamil Machine Translation<br>
> **Hardware:** Dual NVIDIA T4 GPUs (Kaggle Multi-GPU Setup)<br>
> **Code Repository:** [`Transformer_from_scratch`](https://github.com/imrancoder786/ML_FROM_SCRATCH/tree/main/Transformer_from_scratch)<br>

---

## Executive Summary & Motivation 

Before 2017, sequence-to-sequence (Seq2Seq) tasks such as machine translation, summarization, and speech recognition relied heavily on Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and Gated Recurrent Units (GRUs). While effective, recurrent architectures suffered from two fundamental limitations:

1. **Sequential Bottleneck:** Computing hidden state $h_t$ required $h_{t-1}$, preventing parallel execution across sequence lengths during training.
2. **Vanishing/Exploding Gradients:** Capturing long-range dependencies across hundreds of tokens remained challenging despite gating mechanisms.

The landmark paper [*"Attention Is All You Need"*](https://arxiv.org/pdf/1706.03762) (Vaswani et al., 2017) solve the limitations , introducing the **Transformer**—an architecture built exclusively on self-attention mechanisms. 

In this blog, we build the entire Transformer model from scratch using pure PyTorch (`torch.nn`). We will train the architecture on the ['gopi30/'english-tamil'](https://huggingface.co/datasets/gopi30/english-tamil) dataset—a parallel corpus of English-Tamil sentence pairs for English-to-Tamil Translation.

I’ve drawn a lot of inspiration for this blog from [this video](https://youtu.be/ISNdQcPhsts?si=xucQyW8x8Ak_uJQx), which I highly recommend.

Let’s get started!

---

## Architecture Overview

<div align="center">
    <!-- ![Architecture Overview](posts/images/architectue.webp) -->
    <!-- <img src="posts/images/architectue.webp" alt="Architecture Overview" style="width: 90%;"> -->
    <img src="posts/images/architectue.webp" alt="Architecture Overview" width="500">
</div>

It contains 2 macro-blocks:

1.  Encoder
2.  Decoder

and a linear layer.

# Encoder: 
<div align="center">
    <img src="posts/images/encoder.png" alt="encoder" width="300">
</div>

## 1. Embeddings & Sinusoidal Positional Encoding

### 1.1 Input Embeddings with Scaling

<div align="center">
    <img src="posts/images/i_emd.png" alt="input embedding" width="200">
</div>

starts with the input embeddings.


Wait ! **What is an input embedding?**<br>
Embeddings are an array of floating point number. They can be used to represent different modalities such as text, image, video, etc.
Text tokens are discrete indices $x \in \{0, 1, \dots, V-1\}$. we convert each word into an embedding of size 512 (contains 512 floating point numbers). Therefore, we can define our dmodel = 512 .The `InputEmbeddings` module maps each token index to a dense vector space $\mathbb{R}^{d_{\text{model}}}$. 

<div align="center">
    <img src="posts/images/embeddinng.png" alt="input embedding" width="700">
</div>

```python
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embedding weights by sqrt(d_model) as specified in Attention paper .
        return self.embedding(x) * math.sqrt(self.d_model)
```

**Mathematical Rationale for $\sqrt{d_{\text{model}}}$ Scaling:**
As the embedding dimension $d_{\text{model}}$ increases, the variance of the embedding vectors drops. Multiplying by $\sqrt{d_{\text{model}}}$ scales up the embedding values so that when positional encodings are added, the word embeddings maintain an equal order of magnitude.

---

### 1.2 Sinusoidal Positional Encodings

<div align="center">
    <img src="posts/images/pos.png" alt="positional embedding" width="300">
    <img src="posts/images/pos_enc.png" alt="positional embedding" width="700">
</div>

Since self-attention processes all tokens simultaneously without recurrence, the model has no inherent notion of word order. We inject positional information using fixed sinusoidal functions of varying frequencies:

$$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Where:
- $pos$ is the token position in the sequence ($0 \le pos < \text{seq\\_len}$).
- $i$ is the dimension index ($0 \le i < d_{\text{model}}/2$).

**How are position embeddings calculated?**

<div align="center">
    <img src="posts/images/pos_cal.png" alt="positional embedding" width="700">
</div>
For even positions in the position embedding (count starts from 0), we use the 1st formula, and for odd positions in the position embeddings, we use the 2nd formula. We do this for each of the 512 values of a position embedding, for each word/token in the sentence.

So,the position embedding for every position in the sentence is the same, regardless of the sentence. It is the encoder input (sum of embedding and position embedding) that is unique. Therefore, we need to compute the positional encodings only once then
we resue for the every sentence during training & inference.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create positional matrix (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        
        # Calculate div_term in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register buffer so it is saved with state_dict but not treated as a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
```
The sum of the embedding and position embedding gives us the encoder input. For the same word, the vector embedding is the same but the position embedding is different.

**Encoder input = Embedding + Position Embedding**

---

## 2. Multi-Head Self-Attention (MHA) Deep Dive
<div align="center">
    <img src="posts/images/p_multi.png" alt="positional embedding" width="450">
</div>


Before diving into multi-head attention, let’s first understand self-attention with a single head.

### What is self-attention?

Self-Attention allows the network to relate every word in a sequence to every other word.

<div align="center">
    <img src="posts/images/self.png" alt="positional embedding" width="350">
</div>

**The first step** is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we've trained (WQ, WK, WV).

<div align="center">
    <img src="posts/images/q_k_v.png" alt="positional embedding" width="350">
</div>


**We calculate self attention using the formula:**

Given Queries $Q$, Keys $K$, and Values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<div align="center">
    <img src="posts/images/self_soft.png" alt="positional embedding" width="600">
</div>

- Where, sequence length seq = 6, and dmodel = dk = 512.<br>
- The matrices Q (query), K (key) and V (value) are the input sentence of dimension 6x512 (6 rows and 512 columns).

this process is called sigle head.

### now let's move on the Multi-Head Attention:



by Adding the Mutliple-layer of self-attentation together called "Multi-Head Attention".In the *"Attention Is All You Need"* paper they used 8 heads.

<div align="center">
    <img src="posts/images/multi_head.png" alt="positional embedding" width="500">
</div>

### but! why we want to use multiple heads ? we can just use one right ?
let me explain . Yes,one head works.But multiple heads allow the model to learn different kinds of relationships at the same time.

<div align="center">
    <img src="posts/images/map.png" alt="positional embedding" width="700">
</div>

for example,imagin we have only one head .If we’re translating a sentence like "The cat didn't cross the river because it was tired."The word "it" should attend to "cat".
With **one head**, the model has only **one attention pattern** to learn.

```
Head 1

it  ─────────► cat
```

Now consider another sentence:

```
The cat sat on the mat.
```

The word **"sat"** should attend to

-   cat (subject)
-   mat (location)

One head now has to learn **both** relationships.

* * * * *

With multiple heads
-------------------

Suppose we have **4 heads**.

```
Head 1
sat ─────► cat
(subject relationship)

Head 2
sat ─────► mat
(location relationship)

Head 3
sat ─────► on
(grammar)

Head 4
sat ─────► sat
(self information)
```

Each head specializes in something different.


with multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads,so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

<div align="center">
    <img src="posts/images/s_mulit.png" alt="positional embedding" width="600"><br>
    <h6 style="color: #666666;">With multi-headed attention, we maintain separate Q/K/V weight matrices for each head resulting in different Q/K/V matrices. As we did before, we multiply X by the WQ/WK/WV matrices to produce Q/K/V matrices.</h6>
</div>

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices

<div align="center">
    <img src="posts/images/mul_head.png" alt="positional embedding" width="600"><br>
</div>


The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we **concatenate** all these outputs together so the next layer receives **all the information** of the eight martix down into a **single matrix**.

How do we do that? We concat the matrices then multiply them by an additional weights matrix $W^O$.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

**Wailt ! why we multiply them with $W^0$ ?** <br>
Concatenation only stacks the outputs of different heads together. $W^O$ is a learnable linear layer.This learns how to mix those outputs into one meaningful representation so, that the next layer can use.


<div align="center">
    <img src="posts/images/concate.png" alt="positional embedding" width="700"><br>
</div>


Look at the below image.all in one visual of the **Multi-Head-Attention**:

<div align="center">
    <img src="posts/images/full_multi.png" alt="positional embedding" width="700">
</div>


```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by head count h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) # Query Projection
        self.w_k = nn.Linear(d_model, d_model) # Key Projection
        self.w_v = nn.Linear(d_model, d_model) # Value Projection
        self.w_o = nn.Linear(d_model, d_model) # Output Projection

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # 1. Compute Attention Scores: (Batch, h, seq_len, d_k) x (Batch, h, d_k, seq_len) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply Mask (fill 0 positions with -1e9 before softmax)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # 3. Softmax along key sequence length
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # 4. Weighted sum over values: (Batch, h, seq_len, seq_len) x (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model)
        key = self.w_k(k)   # (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model)

        # Reshape into multiple heads: (Batch, seq_len, d_model) -> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Run Scaled Dot-Product Attention
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Concatenate heads back together: (Batch, h, seq_len, d_k) -> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear projection
        return self.w_o(x)
```

---

## 3.Layer Normalization (Add & Norm):

<div align="center">
    <img src="posts/images/norm.png" alt="positional embedding" width="400">
</div>

We normalize the values so that they are in the range of 0 and 1. We also introduce 2 parameters, usually *beta* and *gamma*.

Gamma is multiplicative, we multiply it with the normalized value. Beta is additive, we add beta to the product of gamma and the normalized value.

Beta and gamma introduce some fluctuations in the data as having all the values between 0 and 1 may be too restrictive for the network.

The network will learn to tune beta and gamma to introduce fluctuations when necessary. So, beta and gamma control which values are amplified and by how much.

Unlike Batch Normalization (which normalizes across the batch dimension), **Layer Normalization** standardizes features across the hidden dimension $d_{\text{model}}$ independently for each token:

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$


```python
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Learnable scaling parameter gamma
        self.bias = nn.Parameter(torch.zeros(1))  # Learnable shift parameter beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```
---
## 4. Feed-Forward Networks (FFN):

<div align="center">
    <img src="posts/images/ffn.png" alt="positional embedding" width="450">
</div>

It processes each position in the sequence independently and helps the model to learn complex representations by applying non-linear transformations to the input.

It is a simple, fully connected feed-forward network that is applied to each position separately and identically. It consists of two linear transformations with a ReLU activation in between.

-   **First Linear Transformation**: This layer projects the input into a higher-dimensional space.
-   **ReLU Activation**: A non-linear activation function applied to introduce non-linearity into the model.
-   **Second Linear Transformation**: This layer projects the higher-dimensional representation back to the original dimension

<div align="center">
    <img src="posts/images/ffn_imge.png" alt="positional embedding" width="500">
</div>

Each Transformer block contains a position-wise two-layer Feed-Forward network with a ReLU activation in between:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

```python
class FeedforwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```
---
## 5. Residual Connection

<div align="center">
    <img src="posts/images/resi.png" alt="positional embedding" width="450">
</div>


This allows the model to bypass certain layers by adding the input of a layer directly to its output.

$$ output = \text{f}(X) + X $$

This is known as a skip connection or residual connection and helps in addressing the vanishing gradient problem, facilitating better gradient flow through the network.


```python
class ResidualConnection(nn.Module):
    def __init__(self, dropout :float):
        super().__init__()
        self.dropout =nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward (self , x ,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```
---

# Decoder:

<div align="center">
    <img src="posts/images/decoder.png" alt="positional embedding" width="250">
</div>

This part is similar to the encoder. During training, the target sequence (i.e., the correct output sequence) is used as input to the decoder. However, it is shifted to the right by one position.I

Shifting the target sequence allows the model to predict the next token based on the previous tokens. If the target sequence is [y1, y2, y3, …, yn], it is transformed to [<START>, y1, y2, y3, …, yn-1] before being fed into the decoder.


---


## 6.What is masked multi-head attention?

Our goal is to make the model causal, meaning that the output at a certain position can only depend on the words on the previous positions. The model **must not** be able to see future words.
<div align="center">
    <img src="posts/images/mask.png" alt="positional embedding" width="500">
</div>

We achieve this by replacing all the future words by $- \infty$ in the seq * seq matrices. After the softmax function is applied, all the $- \infty$ will be replaced by 0.


## 7.what is Cross Attention?
<div align="center">
    <img src="posts/images/cross.png" alt="positional embedding" width="500">
</div>


The multi-head attention layer gets keys and values matrices from the encoder’s output and the query from the output of the masked multi-head attention.

As the keys and values are from output of the encoder and the query is from output of the masked multi-head attention, it is **cross attention** (earlier it was self attention).

---

## 8.Linear layer and Softmax:

<div align="center">
    <img src="posts/images/final.png" alt="positional embedding" width="450">
</div>

The decoder stack outputs a vector of floats. **How do we turn that into a word?** That’s the job of the final Linear layer which is followed by a Softmax Layer.

### Linear layer:
The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a **logits vector**.

Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. 

This is achieved using a standard fully connected linear layer, sometimes called a projection layer. Mathematically, if $x_{out} \in \mathbb{R}^{d_{model}}$ is the output vector from the top decoder layer for a specific position, and the vocabulary size is $V$, the linear layer performs:

$$logits = W_{proj} x_{out} + b_{proj}$$


### softmax:


The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

$$P_i = \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$


```python
class ProjectLayer(nn.Module):
    def __init__(self, d_model:int , vocab_size : int ):
        super().__init__()  
        self.proj = nn.Linear(d_model ,vocab_size)

    def forward (self ,x ):
        # (Batxh , seq_lena , d_model) --> (batch , seq_len,vocab_size)
        return torch.log_softmax(self.proj(x) ,dim=-1)
```        




## 9. Assembling Encoder & Decoder Stacks

### 9.1 Encoder Block & Encoder

```python

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedforwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

### 9.2 Decoder Block & Full Transformer Model

```python
class DecoderBlock(nn.Module):
    def __init__(self, self_attentaion_block: MultiHeadAttentionBlock, cross_atteation_block: MultiHeadAttentionBlock, feed_forward_block: FeedforwardBlock, dropout: float):
        super().__init__()
        self.self_attentaion_block = self_attentaion_block
        self.cross_atteation_block = cross_atteation_block
        self.feed_forward_bloack = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. Masked Self-Attention
        x = self.residual_connections[0](x, lambda x: self.self_attentaion_block(x, x, x, tgt_mask))
        # 2. Cross-Attention (Query = Decoder x, Key/Value = Encoder Memory)
        x = self.residual_connections[1](x, lambda x: self.cross_atteation_block(x, encoder_output, encoder_output, src_mask))
        # 3. Feed-Forward Block
        x = self.residual_connections[2](x, self.feed_forward_bloack)
        return x

class Decoder (nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward (self ,x ,encoder_output , src_mask , tgt_mask  ):
        for layer in self.layers:
            x = layer(x , encoder_output ,src_mask,tgt_mask)
        return self.norm(x) 

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        # Full multi-GPU forward pass execution
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        return self.project(decoder_output)
```

Now, we have finished building the transformer from scratch.Let's move on the tranning 

---

## Training a transformer:
Now, we will train our transformer for English-to-Tamil translation. You can use your own preferred language pair instead. I highly recommend trying your own native language. For this project I am using the '['gopi30/'english-tamil'](https://huggingface.co/datasets/gopi30/english-tamil) dataset from Hugging Face. While this dataset contains 5.2 million sentences, our core goal is not to build a flawless translation model. Instead, the objective is to build a transformer from scratch and deeply understand all of its core concepts.Therefore, I am only using 200,000 sentences for training.

Before feeding text into our Transformer, we must convert words into numbers (tokens). However, the model also needs to understand the structural boundaries and properties of sentences. That is where Special Tokens come into play.


```
self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch)
```

<div align="center">
    <img src="posts/images/tranning.png" alt="positional embedding" width="700">
</div>

## Loss function:
We use **Cross-Entropy Loss** to train the model by comparing the predicted probability distribution over tokens with the true target tokens. It works by penalizing the model when it assigns low probability to the correct token, encouraging it to increase confidence in the right predictions. $$( L = -\sum_i y_i \log(p_i) ), where ( y_i )$$ 

 The `ignore_index` option excludes `[PAD]` tokens from the loss calculation, while `label_smoothing=0.1` reduces overconfidence and helps the model generalize better.


I am running the code on Kaggle's dual T4 GPUs using PyTorch's `DataParallel` to utilize both cards. Each epoch takes roughly 1.7 hours, and since I am running 6 epochs, the total training time takes around 11 hours. If you want better results, you can increase the training duration and Dataset. The code also saves checkpoints, allowing you to easily resume training at any time.

### Sample outputs:

```log
SOURCE:Everyday fatal accidents are taking place.
TARGET:ஒவ்வொருநாளும் அரங்கேறும் விபத்துகள் பதறவைக்கின்றன.
PREDICTED:தினமும் விபத்துகள் நடக்கின்றன .
--------------------------------------------------------------------------------
SOURCE:7 people have died already.
TARGET:அத்துடன், 7 பேர் இதுவரை மரணமடைந்துள்ளனர்.
PREDICTED:இதில் 7 பேர் உயிரிழந்தனர் .
processing epoch05: 100%|█████| 5625/5625 [1:50:51<00:00,  1.18s/it, loss=3.802]
--------------------------------------------------------------------------------
SOURCE:The court said the inquiry in the case will continue.
TARGET:இவ்வழக்குத் தொடர்பான விசாரணை தொடர்ந்து நடைபெற்று வருவதாக அமலாக்கத் துறை தெரிவித்தது.
PREDICTED:இந்த வழக்கை விசாரித்த நீதிபதி , விசாரணை தொடரும் என நீதிமன்றம் தெரிவித்தனர் .
--------------------------------------------------------------------------------
SOURCE:Interestingly, India has never lost to Pakistan in the World Cup.
TARGET:உலகக்கோப்பை கிரிக்கெட் வரலாற்றில் இந்தியாவை பாகிஸ்தான் வீழ்த்தியதில்லை என்பது குறிப்பிடத்தக்கது.
PREDICTED:உலகக் கோப்பை இறுதிப் போட்டியில் பாகிஸ்தான் தோல்வி அடைந்தது .
--------------------------------------------------------------------------------
SOURCE:There are bigger fish in the sea.
TARGET:“கடலில் பெரிய பெரிய மீன்கள் இருக்குமே.
PREDICTED:கடல் மிகப் பெரிய காணப்படுகின்றன .

```
## **cross_attention_map:**

<iframe src="posts/cross_attention_map.html" width="100%" height="600px" frameborder="0"></iframe>

---

## Inferencing a Transformer:

The encoder part stays the same as training, <br>
we provide the input: `<SOS>`I love you very much`<EOS>`.

<div align="center">
    <img src="posts/images/inference.png" alt="positional embedding" width="700">
</div>

Instead of providing the entire translation preceded by `<SOS>` and followed by padding `<PAD>` tokens as during training, we provide only `<SOS>` as the decoder input, followed by no padding tokens during inferencing.

As before, we get the key and value matrix from the encoder output and the query matrix from the masked multi-head attention layer's output for the input of the decoder. Then, we get the decoder layer, which is passed through the linear layer and softmax.

The output of the linear layer is known as logits. The softmax function selects a token from our vocabulary corresponding to the position of the token with the maximum value, i.e. the token with the highest probability is selected as the model's predicted output *ti*.

We get the first token (which follows `<SOS>`) in the 1st time step. This first token that we generated is *ti*. While the training happens in 1 time step for each sequence, while inferencing, we need *n* time steps.

after geneating the full transulation "நான் உன்ன மிகவும் நேசிக்கிறேன்" .we get the `<EOS>` token. As we get the `<EOS>` we stop and no more tokens are generated.


That wraps up the deep dive into transformers. I hope you have a better understanding of transformers now! We went through each component of the transformer architecture, understood it’s purpose and saw how everything comes together.

I hope you had a great time reading this. Share an interesting idea or just drop in a Hi at laptop14072024@gmail.com! Until next time 👋
> **Code Repository:** [`Transformer_from_scratch`](https://github.com/imrancoder786/ML_FROM_SCRATCH/tree/main/Transformer_from_scratch)<br>

---
