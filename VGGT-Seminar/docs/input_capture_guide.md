# Input capture guide

## General protocol

- Capture 6–12 photos per scene; start with 8 when practical.
- Keep static scenes unchanged while moving the camera gradually around or through the scene.
- Maintain substantial visual overlap between neighboring images (roughly half or more of the content should remain recognizable).
- Change viewpoint, not digital zoom. Keep the same camera/lens mode.
- Keep portrait/landscape orientation consistent within a baseline sequence.
- Prefer stable exposure and focus for the baseline; create degradations later as derived conditions.
- Use sequential filenames: `001.jpg`, `002.jpg`, `003.jpg`, and so on.
- Preserve original files without editing. Derived images belong under generated outputs, never beside originals.
- Avoid faces, screens with private content, addresses, license plates, or other identifying details when possible.
- Record source, device, scene dynamics, challenges, ground-truth availability, and redistribution permission in `scene_manifest.yaml`.

## Category guidance

### A. Controlled object

Use a LEGO vehicle/model or another rigid object on a textured surface. Circle it in small steps with stable lighting. Expected behavior: strong overlap and a favorable reconstruction case.

### B. Indoor room or desk

Capture a desk, shelf, room corner, or computer setup with foreground/background depth. Translate gradually; include planar surfaces and mixed textures.

### C. Outdoor static scene

Use a facade, courtyard, parked vehicle, or statue. Keep moving objects out of the central subject and avoid large time/exposure changes.

### D. Textureless or repetitive scene

Include a plain wall/table, tiles, blinds, or repetitive windows while retaining some textured context for orientation. This intentionally weakens correspondence cues.

### E. Reflective or transparent scene

Use glass, a monitor, glossy parts, windows, or metal. Reflections should change with viewpoint; document them as expected violations of simple geometry.

### F. Dynamic scene

Keep the camera path controlled while a person/toy/screen changes between frames. Obtain consent and avoid identifiable faces. This intentionally violates the static-scene assumption.

### G. Degraded input

First capture a good static baseline. Generate blur, reduced brightness, JPEG compression, reduced resolution, reduced overlap, and missing-intermediate-view conditions reproducibly from copies in the output directory.

## Before running

1. Copy originals into the appropriate `data/custom_inputs/<scene>/` folder.
2. Complete a `scene_manifest.yaml` from `data/custom_inputs/scene_manifest.template.yaml`.
3. Verify filenames and order.
4. Visually check that images show one scene and contain no unintended private information.
5. Keep the manifest's `ordered_images` explicit; glob order alone is not experimental evidence.
