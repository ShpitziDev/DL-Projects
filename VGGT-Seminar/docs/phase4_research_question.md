# Phase 4 research question

## Central question

> How robust is the pretrained VGGT model when reconstructing geometry and camera relationships from different real-world and non-ideal multi-view image sets?

This question follows VGGT's intended use: one or more RGB views of the same scene are processed jointly to predict cameras, depth, point maps, confidence, and tracks. The study focuses on pretrained inference, not reproduction of the original 160k-iteration, 64-A100 training procedure.

## Supporting questions

1. How does reconstruction behavior change with the number of views?
2. How important is overlap between neighboring images?
3. How sensitive are predictions to image order and the first reference frame?
4. What happens under blur, low light, reflections, textureless surfaces, repetitive patterns, and moving objects?
5. How different are directly predicted point maps from depth-unprojected point clouds?
6. How stable are predicted relative camera poses and intrinsics across controlled variants?
7. Where do confidence maps agree with visible uncertainty or inconsistency?
8. Can VGGT produce useful geometry outside ideal static, overlapping-view conditions?

## Evidence boundaries

- **Paper reproduction:** limited to using the official architecture, pretrained checkpoint, preprocessing, and output definitions. We do not reproduce training or complete benchmark tables.
- **Our experiment:** controlled comparisons on locally captured scenes using a fixed code/checkpoint pin and recorded conditions.
- **Qualitative analysis:** visible geometric coherence, object/plane shape, discontinuities, floaters, camera layout, failure localization, and confidence-map agreement.
- **Quantitative analysis:** only measurements defined from available evidence, such as runtime/VRAM, output finiteness, confidence summaries, cross-condition camera stability after gauge-aware comparison, and direct-versus-unprojected point disagreement. These are diagnostic metrics, not benchmark accuracy.
- **Unsupported claims:** without ground truth, we will not call a reconstruction metrically accurate, assign absolute pose/depth error, or claim exact reproduction or superiority over another method.

## Hypotheses

- High-overlap, static, textured scenes will produce the most coherent geometry and camera layouts.
- Reducing views or removing intermediate overlap will increase disagreement and lower confidence around ambiguous regions.
- Image order should preserve scene relationships after alignment, but changing the first frame may alter the learned canonical frame and scale.
- Reflective, transparent, dynamic, and texture-poor regions will show localized geometric artifacts; useful confidence maps should flag at least some of them.
- Depth-unprojected geometry may be more internally camera-consistent than the direct point-map branch, while both can share appearance-driven failures.

## Planned study size

The framework supports seven categories. The first study should use 3–5 carefully captured scenes, not every category at once. A recommended first set is controlled object, indoor scene, one difficult material/texture scene, and one degraded variant family.
