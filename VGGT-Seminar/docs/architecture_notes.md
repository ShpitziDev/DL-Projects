# Architecture notes

```text
RGB views
   -> DINOv2 patch tokens + camera/register tokens
   -> alternating frame-wise/global transformer aggregator
      -> camera tokens -> iterative camera head -> intrinsics/extrinsics
      -> intermediate image tokens -> DPT depth head -> depth + confidence
      -> intermediate image tokens -> DPT point head -> world points + confidence
      -> dense features + queries -> tracking module -> tracks + visibility
```

The first camera defines the canonical reference frame used in supervision. Geometry outputs must be retained independently: a visualization derived from depth and cameras is not interchangeable with the direct point-map branch. Future wrappers should serialize raw tensors with shape/dtype/unit/convention metadata before producing visualizations.

Open code audit items: token layout, exact layer count terminology, feature width/head count, DPT taps, pose encoding convention, confidence transform, preprocessing shape policy, track query coordinate convention, and current memory-retention behavior.

## Reconciliation at pinned commit `a288dd0` (Official code)

- Default image size 518, patch size 14, embedding width 1024, 16 attention heads.
- DINOv2 ViT-L/14 with four register tokens is the default patch embedder.
- `depth=24` creates 24 frame blocks and 24 global blocks, executed as 24 alternating frame/global pairs. Thus “24 alternating-attention blocks” in prose corresponds to 48 individual attention modules in code.
- Two parameter banks exist for camera and register tokens: one used only by the first frame and one shared by remaining frames. Each frame receives one camera token plus four register tokens; patches start at index 5.
- Cached aggregator/DPT layers are `[4, 11, 17, 23]`; cached frame and global outputs are concatenated to width 2048.
- Camera head uses the final camera token, four transformer trunk blocks, four iterative refinements, and 9D `absT_quaR_FoV` encoding.
- Point DPT emits 3D coordinates plus confidence; depth DPT emits depth plus confidence. Both chunk dense decoding at eight frames by default in current code.
- Tracking is optional at call time: head exists by default, outputs tracks, visibility, and confidence only when query points are supplied.
- Attention defaults to PyTorch fused scaled-dot-product attention; `flash_attn` is not required.
- Aggregator inference caches only requested layer outputs and explicitly deletes per-step intermediates, reflecting the May 2026 memory fix.
