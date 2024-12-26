---
title: 'The Evolution of Data Parallelism: From DP to DeepSpeed and FSDP'
date: 2024-03-20
permalink: /posts/2024-03-20-data_parallelism/
tags:
  - Machine Learning
  - Deep Learning
  - Training Framework
---

# Data Parallelism Overview
Data Parallel (DP) is the foundational and most intuitive form of parallelism in model training. It replicates the model across multiple workers, with each worker processing a subset of the data batch.

Distributed Data Parallel (DDP) enhances DP by launching multiple processes, where each process handles a mini-batch of data independently. This approach not only speeds up training but also reduces communication overhead through efficient algorithms like ring-reduce.

**Parameter Server Architecture**: Traditionally, even though data (and intermediate results, like logits) are distributed, the model itself remains replicated across GPUs. The challenge then becomes managing the model and its associated data efficiently.

**Optimization Techniques**:
1. **Optimizer States**: DeepSpeed Zero optimizes this by only gathering optimizer states during the backward pass, which are not needed during the forward pass.
2. **Gradients**: The second stage involves partitioning gradients and optimizing communication, allowing each partition to update its corresponding optimizer states locally.
3. **Model Weights**: The third stage partitions model weights, which are used in the forward pass, requiring additional communications (e.g., all-gather) during training.

FSDP (Fully Sharded Data Parallel) builds on the principles of Zero's third stage but with further optimizations that enhance performance, making it suitable for training large language models (LLMs) more efficiently.

Look out for the second blog in this series, where we delve deeper into DeepSpeed and FSDP's practical applications.



# Deep Dive on DeepSpeed and FSDP
The following content assumed you have understanding about GPU communication operations.

## DeepSpeed stages and GPU communication volumes
* stage1: $O(3\theta)$ optimizer weights.
* stage2: $O(2\theta)$ optimizer weights + gradients.
* stage3: $O(3\theta)$ optimizer weights + gradients + model weights.

## FSDP unit
The smallest partition and communication unit in FSDP is turned into FlatParameter.

## FlatParameter
Imagine you want to flatten a MLP, `y=w2(act(w1(x)))`. `w1` has the dimension `h x 4h` and `w2` has the dimension `4h x h`. After flattening, the FlatParameter has the shape, `1 x (h x 4h +4h x h)`. We let FSDP unit to be the MLP, and we want to convert it into FlatParameter. The reason for FlatParameter is it's easier to split based on the number of devices.

To make it more intuitive, let `h=1`. We have the following

[Graph]


Why FlatParameter?
* spatial locality
* communication: brandwidth utilization, memory buffer?
* simplified API for operation:
* simplified representation: less metadata.
* input-agnostic:
* device-agnostic:

## Sharding




## Q & A

#### Q1: How does FSDP compare to Tensor Parallelism? 
There could be some common misunderstanding that FSDP is similar to Tensor Parallelism because they both partition model parameters and distribute across GPUs. However, they differ a lot if you really take a closer look at them.
* **Fundamental difference**: FSDP is essentially data parallelism, and the model parameter sharding across GPUs is just a extra "feature" evolved to reduce model redundency. But TP by nature slices model weights and distribute across GPUs.
* **Input weights and "splitting" methods difference**: Sharding is the process of flattening and partitioning, and it's model weight agnostic. In other words, it can shard any number of weights together into one FlatParameter and then partition based on the number of GPUs. However, TP needs to slice model weights along column dimension or row dimension depending on the matrix weight operation to make sure each GPU can compute logits more independently to reduce communication. In the MLP example above, we need to slice `w1` along row dimension and `w2` column dimension.
* **GPU persistence difference**: During FSDP training, on each GPU, sharded weights are prefetched upon usage and disposed after usage. Therefore, FSDP doesn't persist full model weights on each GPU. However, for TP, model weights are partitioned, and they persist across GPUs during training. (Another view is that FSDP persists a shard of model parameter on each GPU.)


#### Q2: How does FSDP compare to model parallelism?
Similar to Q1, they differ from these three perspectives. The second point worth mentioning, MP partitions model weights inter-layer, and TP partitions model weights intra-layer.



#### Q3: diff between zero3 and fsdp despite their are very similar?
> The FSDP algorithm is motivated by the
ZeroRedundancyOptimizer [27, 28] technique from DeepSpeed but
with a revised design and implementation that is aligned with the
other components of PyTorch. FSDP breaks down a model instance
into smaller units and then flattens and shards all of the parameters
within each unit. The sharded parameters are communicated and
recovered on-demand before computations, and then they are immediately discarded afterwards. This approach ensures that FSDP
only needs to materialize parameters from one unit at a time, which
significantly reduces peak memory consumption

On the one hand, it's more integrated with PyTorch ecosystem. On the other hand, FSDP unit is much smaller than Zero3 which makes model weight fetching more streamlined and efficient w.r.t computation.
<!-- #### Q4: async gradient update + traffic aware routing 
* does backward contain a sequence of gradient computation O(2\theta) but forward only involves one-time logits aggregation? -->
