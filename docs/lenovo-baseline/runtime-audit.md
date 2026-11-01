# Why the Lenovo baseline takes about 67 hours

Audit date: 1 November 2026, Asia/Shanghai. This is a source/configuration audit,
not a new training result or a measured optimization benchmark.

## Finding

The 67-hour estimate is consistent with the observed training workload. The
configuration is not an established CIFAR-10 reproduction recipe. Copying the
training equations did not validate the model size, batch/LR choice, or budget
for an 8-GB laptop GPU. Those choices should have been justified before launch.

The active training loop is upstream-derived. Direct comparison found one
forward/backward/optimizer/EMA update per minibatch, not an added repeated
training loop. The supervisor starts the upstream entry point once and polls
it every 30 seconds; it does not train a model.

## Published recipe versus this run

The source is [official JiT commit cbc743a2](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17).
The [published training command](https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/README.md#training)
uses ImageNet and eight H200 GPUs. There is no published CIFAR recipe in this repository.

| Setting | Official B/16 example | Current Lenovo run |
|---|---|---|
| Dataset / resolution | ImageNet, 256 x 256 | CIFAR-10, 32 x 32 |
| Model body | 12 blocks, width 768, 12 heads | Same B body, approximately 130M total parameters |
| Patch size / image tokens | 16 / 256 | 2 / 256 |
| Context tokens | 32, inserted after four blocks | Same |
| GPUs / global batch | 8 H200 / 1024 | 1 RTX 4060 Laptop / 8 |
| Budget | 600 epochs in README | 200 epochs, also the upstream parser default |
| Learning rate | blr=5e-5, actual LR=2e-4 at batch 1024 | Explicit LR=1e-4 |
| Compilation | Block and output forward use torch.compile | TORCH_COMPILE_DISABLE=1 |
| Conditioning | Class conditional with CFG | Unconditional, label dropout 1, CFG 0 |

The first four blocks process 256 tokens and the remaining eight process 288.
Thus reducing image resolution did not reduce the dominant Transformer-body
work. Patch embedding and output projection are smaller, but this is not a
small CIFAR model. Also, the original 768-to-128 patch projection becomes
12-to-128 at patch size 2; it no longer has the same input bottleneck interpretation.

## Measured cost and limits

Two completed epochs took 20:18 and 20:14. The median of the last 100 printed
step-time rows in the earlier copied log was 0.1926 seconds/update, approximately
41.5 images/second. The live audit again observed about 0.193 seconds/update,
GPU utilization 99%, approximately 6.2 GiB resident memory, and near-zero loader
wait. This does not establish optimal kernel efficiency.

```text
50,000 images / batch 8 = 6,250 updates per epoch
6,250 x 200 = 1,250,000 optimizer updates
1,250,000 x 0.1926 / 3,600 = 66.875 training hours
```

Evaluation, checkpoint writes and interruptions add wall time. No measured
profile yet attributes the 0.193 seconds between forward, backward, Adam and
EMA. Increasing batch size does not divide wall time by the batch-size ratio:
larger batches also cost more per update. Gradient accumulation similarly
preserves the need to process every microbatch.

The earlier batch-size decision used only eight updates per candidate, including
startup. It demonstrated capacity, not the best steady throughput. Batch 16
reached 7.38 GiB allocated and was slower in that short probe; this is a reason
for caution, not a definitive optimal-batch result.

## Recipe and implementation issues

1. **Unvalidated optimization budget.** 200 epochs is a parser default and a
   local allocation, not evidence that this model reaches FID below 50 on CIFAR.
   Five warmup epochs mean 31,250 updates here. The default EMA1 half-life is
   about 6,931 updates, or 55,449 image presentations at batch 8. Equal EMA decay
   per update does not preserve its image-based horizon across batch sizes.
2. **LR does not follow the author's scaling rule.** That rule would give
   5e-5 x 8 / 256 = 1.5625e-6 at batch 8; the override is 64 times that value.
   This does not prove 1e-4 is wrong, nor justify blindly reducing it 64-fold.
   It establishes that this is a local unvalidated recipe rather than a copied
   published training configuration.
3. **Compilation is disabled.** The launch record explicitly disables the
   author's compiled block path. Its performance penalty is unmeasured. The
   official attention also explicitly forms FP32 QK scores and materializes
   attention weights; it does not call PyTorch fused SDPA. These implementation
   details are inherited from upstream, not newly written attention code.
   A different kernel changes numerical behavior and requires comparison.
4. **Unconditional sampling has redundant work.** The copied sampler computes
   conditional and null-label predictions even when CFG=0. With the current
   interval and finite outputs, only the null-label prediction contributes.
   Fifty Heun intervals with the final Euler step require 99 field evaluations,
   currently 198 model forwards per generated batch. Removing the unused branch
   can halve this forward count; it does not halve training time. No such change
   was applied to the running source, and no speedup has been measured.

The training path uses a single denoiser prediction, not the 50-step sampler.
The redundant CFG branch therefore cannot explain the 67-hour training estimate.

## Recommended next decision

Do not treat completion of 200 epochs as the default proof that the baseline is
valid. Preserve the latest checkpoint, then use a bounded performance check
before committing the rest of the budget. With the GPU free, the check should
reuse the official model/objective/optimizer: warm up, time at least 100 updates,
and separate forward/backward/Adam/EMA costs at safe physical batch sizes.
Only a demonstrated equivalent execution improvement should enter a resumed run.

If the full B body remains too expensive, explicitly choose between a smaller
Transformer experiment and the full-model budget. A smaller body or fewer
tokens is a new architecture configuration; it must establish its own baseline
quality. Simply shortening epochs cannot promise convergence. Adding gradient
accumulation cannot promise a proportional speedup.

For context, [Meta's official CIFAR flow-matching example](https://github.com/facebookresearch/flow_matching/blob/main/examples/image/README.md#results)
reports an unconditional U-Net result at 1,800 epochs, with a 3,000-epoch launch
cap. That supports neither a short wall-time guarantee nor this Transformer
recipe. Replacing the experiment with its U-Net would change the FFN research
question, so no such substitution was made.

## Verification and preserved state

The live launch uses frozen commit 3b07a57. Hashes bind 12 source/configuration
files to that launch. AST comparison confirmed unchanged Denoiser loss,
generation, sampling steps and EMA, plus native SwiGLU/block/output forwards.
The epoch-loop diff was inspected directly. See [runtime-audit.json](runtime-audit.json).

This audit changes documentation only. Training code, configuration, running
process, environment, checkpoints, Report1 and Report2 were not edited or
restarted. No new GPU workload was launched beside the occupied device.
