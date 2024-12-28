---
title: 'The Evolution of Data Parallelism: From DP to DeepSpeed and FSDP'
date: 2024-03-20
permalink: /posts/2024-03-20-data_parallelism/
tags:
  - Machine Learning
  - Deep Learning
  - Training Framework
  - Data Parallelism
---

# Data Parallelism Overview
Here I am trying to illustrate all DP techniques chronically with reasoning for each improvement.

1. Data Parallel (DP) is the foundational and most intuitive form of parallelism in model training. It replicates the model across multiple workers, with each worker processing a subset of the data batch.

   * Parameter Server(PS) Architecture: DP achieves multiple workers computing and synchronizing gradients by parameter server. In practice, if you training model with DP on two gpus but they occupy uneven amount of memories. The reason is that one of two GPUs (typically the first GPU) acts as a parameter server stores and updates optimizer states. See the left graph below.
   * During gradient communication, each worker (on one thread) is responsible for `push` and `pull` gradients at the same time to/from PS as shown on the right graph. Workers(single process multiple threads) contribute to the process actively.

![DP](../../images/blogs/dp.png)
However, as the number of workers grow, the instance communication volume upon push and pull gradients incur a sudden **communication overhead** which becomes a bottleneck for training speed.

1. Distributed Data Parallel (DDP) enhances DP by launching multiple processes, where each process handles a mini-batch of data but maintain optimizer states independently. This approach not only speeds up training but also reduces peak communication overhead through efficient all-reduce communication operation.
   * No PS. In other words, all workers (multiple processes) on all GPUs participate gradients communication actively.
   * During all-reduce communication, each GPU requires `send` and `rev` operation in a streaming manner, and this is enabled by multi-process. 

![DDP](../../images/blogs/ddp.png)

Despite the communication improvement from DP to DDP, model and optimizer states remains replicated across GPUs. On the one hand, it enables simple and efficient optimization. On the other hand, such a replication lead to a lot of **redundant memory usage**. Can we optimize the redundancy? Deepspeed Zero is proposed to solve such a problem.





1. Deepspeed Zero: DeepSpeed has three stages, and it optimize the redundancy incrementally from the first stage to the third stage.
   1. **Optimizer States(OS)**: During DP forward pass, optimizer states are not needed because there is no gradient update. Can we partition the optimizer states during forward and gather and update it during backward? Zero stage1 handles this situation exactly -- with each GPUs hosts partial optimizer states, after all-reduce gradients, it all-gather optimizer states to sync weights. ![zero1](../../images/blogs/deepspeed_zero1.png)
   2. OS + **Gradients(G)**: If you think deeply, each partial optimizer states only requires partial average gradients. It means we can also partition gradients across GPUs and only reduce-scatter gradients such that each partial gradients correspond to each partial optimizer state it corresponds to. Therefore, it not only optimizes memory usage (on gradients) but also communication from $3\theta$ to $2\theta$. ![zero2](../../images/blogs/deepspeed_zero2.png)
   3. OS + G + **Model Weights**: So far we have optimized memory usage of tensors related to backward, how about tensors used for forward pass, the model parameter? In DP forward pass, each worker do forward computation independently with full model parameters, and it sounds tricky that how it can be optimized. Actually, the idea behind model weights optimization is "dispose it after usage". Suppose we have three model layers and two GPUs. $W_{xy}$ denotes xth layer's yth partition. For example, $W_{10}$ means the second layer weight on the first GPU. Suppose we forward pass the second layer on first GPU, we need to all-gather $W_{10}$ on local gpu and $W_{11}$ on the second GPU to materialize the second layer $W_{1}$. Afterwards, it discards $W_{1}$ and repeat the layer materialization step described. Therefore, the peak memory usage on each GPU (for model parameter only) is only the one layer weight in this example. ![zero3](../../images/blogs/deepspeed_zero3.png)

In practice, deepspeed zero3 could be significantly slower than DDP as the model is very large. The root cause is that the granularity of model weights sharding is still not small enough to fully utilize compute-communication overlapping.


Here is a snippet from FSDP paper:
> The FSDP algorithm is motivated by the
ZeroRedundancyOptimizer [27, 28] technique from DeepSpeed but
with a revised design and implementation that is aligned with the
other components of PyTorch. FSDP breaks down a model instance
into smaller units and then flattens and shards all of the parameters
within each unit. The sharded parameters are communicated and
recovered on-demand before computations, and then they are immediately discarded afterwards. This approach ensures that FSDP
only needs to materialize parameters from one unit at a time, which
significantly reduces peak memory consumption




4. FSDP (Fully Sharded Data Parallel) builds on the principle of Zero's third stage but with further optimizations that enhance performance, making it suitable for training large language models (LLMs) more efficiently. It proposes FSDP unit which is the smallest unit in FSDP that is turned into `FlatParameter` before sharding and materialized upon computation.
On the one hand, it's more integrated with PyTorch ecosystem. On the other hand, FSDP unit is smaller than its equivalent "partition unit" in Zero3 which makes model weight fetching more streamlined and efficient w.r.t computation.




## Q & A

#### Q1: How does FSDP compare to Tensor Parallelism? 
There could be some common misunderstanding that FSDP is similar to Tensor Parallelism because they both partition model parameters and distribute across GPUs. However, they differ a lot if you really take a closer look at them.
* **Fundamental difference**: FSDP is essentially data parallelism, and the model parameter sharding across GPUs is just a extra "feature" evolved to reduce model redundency. But TP by nature slices model weights and distributes across GPUs.
* **Input weights and "splitting" methods difference**: Sharding is the process of flattening and partitioning, and it's model weight agnostic. In other words, it can shard any number of weights together into one FlatParameter and then partition based on the number of GPUs. However, TP needs to slice model weights along column dimension or row dimension depending on the matrix weight operation to make sure each GPU can compute logits more independently to reduce communication. In the MLP example above, we need to slice `w1` along row dimension and `w2` column dimension.
* **GPU persistence difference**: During FSDP training, on each GPU, sharded weights are prefetched upon usage and disposed after usage. Therefore, FSDP doesn't persist full model weights on each GPU. However, for TP, model weights are partitioned, and they persist across GPUs during training. (Another view is that FSDP persists a shard of model parameter on each GPU.)


#### Q2: How does FSDP compare to model parallelism?
Similar to Q1, they differ from these three perspectives. The second point worth mentioning, MP partitions model weights inter-layer, and TP partitions model weights intra-layer.
