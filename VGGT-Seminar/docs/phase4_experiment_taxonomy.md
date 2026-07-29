# Phase 4 experiment taxonomy

This taxonomy turns the central research question into controlled comparisons. It is a local evaluation of the pretrained checkpoint, not a reproduction of every VGGT paper benchmark.

## Scene families

| ID | Scene family | Primary stressor | Capture target | Main observation |
|---|---|---|---|---|
| A | Controlled object | Baseline multi-view geometry | 6–12 overlapping views around one rigid object | Coherent shape, camera arc, confidence |
| B | Indoor scene | Depth range, occlusion, repeated structure | Gradual sweep with near and far surfaces | Layout consistency and holes |
| C | Outdoor scene | Large depth range and illumination | Stable exposure where possible; overlapping viewpoints | Far-depth stability and camera relation |
| D | Textureless scene | Weak correspondence cues | Plain walls or matte surfaces with surrounding context | Confidence loss and surface instability |
| E | Reflective scene | View-dependent appearance | Reflective object plus static textured context | Floaters, duplicated surfaces, confidence |
| F | Dynamic scene | Violation of static-scene assumption | Static camera path with one moving element | Local versus global corruption |
| G | Derived degradation | Blur, darkness, resolution, or compression | Generated only from an original captured scene | Change relative to the same original views |

## Controlled variables

Change one axis at a time while holding the source scene fixed:

- **Frame count:** 2, 4, 6, and 8 evenly spaced frames, limited by available images.
- **Order:** original, reversed, rotate-first, and deterministic shuffled order.
- **Degradation:** none, blur, low light, low resolution, and JPEG compression.
- **Scene family:** compare only cautiously because content, motion, and capture geometry also change.

The default notebook constructs a small selected plan. `BUILD_FULL_PLAN=True` only expands the plan; it does not approve inference.

## Condition identity and provenance

Each condition should retain:

- scene manifest and ordered source paths;
- file hashes and selected indices;
- frame count, order variant, degradation, and random seed;
- checkpoint hash, code revision, device, and numeric precision;
- output schema and timing information.

Use a condition ID shaped like `<scene>__n<count>__<order>__<degradation>`. Never overwrite an earlier run; use a timestamped run directory.

## Observation rubric

For every completed condition, record observations before comparing conditions:

1. **Camera relationship:** Does the recovered path resemble the physical capture movement? Are there flips, jumps, or collapsed baselines?
2. **Geometry:** Are major surfaces coherent? Note holes, floaters, duplication, warped planes, and foreground/background leakage.
3. **Confidence:** Report distribution summaries and inspect whether low-confidence regions correspond to visible failures.
4. **Tracking:** If enabled, inspect continuity and obvious drift; do not treat track count alone as accuracy.
5. **Sensitivity:** Describe the change from the matched baseline, not merely whether an output looks plausible.

Without ground-truth cameras or geometry, camera and reconstruction quality remain qualitative. Confidence is a model output, not proof of correctness. Cross-scene differences are descriptive and cannot isolate a single causal factor.

