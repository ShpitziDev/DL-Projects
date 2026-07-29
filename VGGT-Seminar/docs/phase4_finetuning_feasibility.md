# Phase 4 fine-tuning feasibility

## Current decision: do not fine-tune yet

The lecturer makes fine-tuning optional. A scientifically meaningful adaptation requires a demonstrated, repeated failure; a target metric; legally usable training/validation data; and an intervention likely to address the failure. None exists before the Phase 4 inference study.

## Technical feasibility

- Official current code includes a post-publication training reimplementation and a Co3D example.
- Its default fine-tuning setup freezes the 1.2B-parameter aggregator and can train camera/depth heads without tracking.
- Documentation assumes multi-GPU DDP, while this machine has one RTX 5080 with about 16 GB VRAM.
- Phase 3 inference peaked near 5.35 GiB for one image, but training requires gradients, saved activations, optimizer state, and data batches; inference memory does not establish training feasibility.
- Gradient checkpointing, smaller per-GPU image counts, gradient accumulation, head-only training, and bfloat16 can reduce memory. They do not guarantee a useful or stable experiment.
- LoRA is not automatically appropriate: the official training path does not establish a LoRA target/recipe, and adapting attention without a strong hypothesis would add implementation risk.

## Promotion gate

Fine-tuning may be proposed only if all are true:

1. At least one failure recurs across multiple captured scenes or controlled variants.
2. A held-out evaluation measure exists.
3. The intended trainable module is linked to the failure hypothesis.
4. Data permission and train/validation separation are documented.
5. A no-training baseline and simpler alternatives are recorded.
6. A one-batch memory probe succeeds with an approved stop budget.
7. The plan specifies what outcome would falsify the adaptation hypothesis.

## Most defensible candidate

If Phase 4 reveals a narrow, repeated domain failure, the first candidate is frozen-aggregator adaptation of a small depth or camera head on a small custom domain. Tracking-head adaptation is a separate task and should not be mixed into the same experiment. Full-backbone training and training-from-scratch remain out of scope.
