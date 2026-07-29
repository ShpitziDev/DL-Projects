# Dependency compatibility

Pinned code: `a288dd0f14786c93483e45524328726ab7b1b4ce`. Selected versions are recorded in `requirements/`.

| Dependency | Official pin/spec | Selected | Windows | Scope and rationale |
|---|---:|---:|---|---|
| Python | >=3.10 | 3.11.15 | Yes | Stable midpoint supported by project and Torch |
| torch | 2.3.1 | 2.13.0+cu130 | Yes | Historical pin predates Blackwell; current stable CUDA 13.0 build |
| torchvision | 0.18.1 | 0.28.0+cu130 | Yes | Required matching Torch release |
| numpy | 1.26.1 / `<2` | 1.26.4 | Yes | Preserves upstream NumPy 1.x constraint |
| Pillow | unpinned | 12.2.0 | Yes | Core preprocessing; wheel already selected with TorchVision |
| huggingface_hub | unpinned | 0.34.4 | Yes | Model mixin/loading; imports tested offline |
| einops | unpinned | 0.8.1 | Yes | Core tensor rearrangement |
| safetensors | unpinned | 0.6.2 | Yes | Core package metadata/loading support |
| opencv-python | pyproject only | 4.11.0.86 | Yes | Geometry/image utilities; compatible with NumPy 1.26 |
| flash_attn | absent | not installed | Poor | Not required; code uses PyTorch SDPA |
| gradio/viser | exact demo pins | not installed | Likely | Demo only; excluded from Phase 2 |
| pycolmap/pyceres | demo pins | not installed | Risk | COLMAP/BA only; Windows wheels/behavior require separate audit |
| LightGlue | unpinned Git URL | not installed | Risk | Optional BA/demo dependency; not reproducibly pinned upstream |
| Hydra/OmegaConf | demo/training | not installed | Yes | Training/demo only |

## Groups

- Project core/dev: `requirements/base.txt`
- GPU foundation: `requirements/pytorch-cu130.txt`
- VGGT core inference: `requirements/vggt-inference.txt`
- Demo: `requirements/optional-demo.txt`
- Evaluation/BA: `requirements/optional-evaluation.txt`
- Training/fine-tuning: `requirements/optional-training.txt`

The official `requirements.txt` was not installed. VGGT itself was installed editable with `--no-deps`, after installing the reconciled core list. Optional CUDA extensions were neither installed nor compiled.
