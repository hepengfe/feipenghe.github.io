---
title: 'Training with Parallelism: Insights and Techniques'
date: 2024-06-20
permalink: /posts/2024-03-20-training_with_parallelism_1/
tags:
  - Machine Learning
  - Deep Learning
  - Training Framework
---

This blog post illustrates my understanding of the development of parallelism in a practical context. I aim to omit unnecessary details and present the information concisely, logically, and intuitively. Although I am not a systems expert, these concepts are frequently discussed in training tasks and interviews.

Back in 2021, before graduating from the University of Washington, training frameworks were rudimentary, primarily used for training small encoder or encoder-decoder models for specialized tasks. As time progressed, multi-task learning and scaling laws have driven increases in model sizes, necessitating the evolution of training frameworks to accommodate increasingly large models. There are primarily two methods to reduce computation and memory needs: 
1. **Quantization**: This method reduces bit width, decreasing compute and memory requirements. However, despite promising results, quantization often slightly degrades model performance.
2. **Parallelism**: This approach involves splitting matrices (data or model weights) across nodes/GPUs to facilitate larger model training.

My approach to understanding includes considering the target partitioning dimension as the first dimension and abstracting all other dimensions for 2D visualization.

**Data Parallelism**: This method involves sharding model and optimizer weights across bs x other dimensions. Note that despite FSDP sharding model and optimizer weights, it is fundamentally a data parallel method rather than model parallelism. Vanilla DP suffers from model weight redundancy, and its variants (DeepSpeed, FSDP) address this by sharding weights, grabbing them upon use, and disposing of them afterwards. For a more detailed discussion, please refer to another blog post.

**Model and Pipeline Parallelism**: This form of parallelism does not split matrices but distributes entire layers across GPUs and nodes. It can be seen as a basic form of pipeline parallelism with a `minibatch=1`. I provide an example graph of `minibatch_size=1`, `minibatch_size=3`, and a computation of the bubble rate.

**Tensor Parallelism**: This method splits the model weight matrix. A representative training framework is Megatron, which not only reduces memory needs but also minimizes communication overhead based on Transformer computation. I conceptualize this as the input data shape being `1 x 1`, indicating a batch size and hidden dimension of 1. This simplification helps focus on tensor parallelism. Note that the first dimension of the final output is still `1`, and the second dimension depends on the task. For simplicity, consider a regression task where the output dimension is also `1`. (requires reduction)

**Sequence and Context Parallelism**: This involves partitioning seq x other dims. With tensor parallelism already partitioning self-attention and non-linear layers, it is natural to also partition normalization layers, as tensor parallelism does not cover logits around these layers. (Note: the dropout layer is also partitioned; please refer to the SP and CP blog for more details)

Each device stores a partial sequence of Q, K, V projection matrices. However, each device only computes its partial Q. Consider a `seq_len=3` with each GPU storing only one token's logits:

**Logits Storage**:
- GPU1: Q1, K1, V1
- GPU2: Q2, K2, V2
- GPU3: Q3, K3, V3

**Computation**:
- GPU1 computes `SelfAttention(Q1, K1, V1)`, `SelfAttention(Q2, K2, V2)` and `SelfAttention(Q3, K3, V3)` by fetching K2, V2, K3, V3 from GPU2 and GPU3. This is because modern LLM architectures typically use multiple Qs sharing the same K, V, thus reducing the need for K, V communication.
