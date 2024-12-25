---
title: 'The Evolution of Data Parallelism: From DP to DeepSpeed and FSDP'
date: 2024-03-20
permalink: /posts/2024-03-20-data_parallelism/
tags:
  - Machine Learning
  - Deep Learning
  - Training Framework
---

Data Parallel (DP) is the foundational and most intuitive form of parallelism in model training. It replicates the model across multiple workers, with each worker processing a subset of the data batch.

Distributed Data Parallel (DDP) enhances DP by launching multiple processes, where each process handles a mini-batch of data independently. This approach not only speeds up training but also reduces communication overhead through efficient algorithms like ring-reduce.

**Parameter Server Architecture**: Traditionally, even though data (and intermediate results, like logits) are distributed, the model itself remains replicated across GPUs. The challenge then becomes managing the model and its associated data efficiently.

**Optimization Techniques**:
1. **Optimizer States**: DeepSpeed Zero optimizes this by only gathering optimizer states during the backward pass, which are not needed during the forward pass.
2. **Gradients**: The second stage involves partitioning gradients and optimizing communication, allowing each partition to update its corresponding optimizer states locally.
3. **Model Weights**: The third stage partitions model weights, which are used in the forward pass, requiring additional communications (e.g., all-gather) during training.

FSDP (Fully Sharded Data Parallel) builds on the principles of Zero's third stage but with further optimizations that enhance performance, making it suitable for training large language models (LLMs) more efficiently.

Look out for the second blog in this series, where we delve deeper into DeepSpeed and FSDP's practical applications.
