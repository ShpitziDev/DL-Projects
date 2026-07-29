# Custom multi-view inputs

These directories hold user-captured originals. No scene imagery is included by default and the official cartoon sanity-check image must not be copied here.

Available categories:

- `controlled_object/`
- `indoor_scene/`
- `outdoor_scene/`
- `textureless_scene/`
- `reflective_scene/`
- `dynamic_scene/`

Copy `scene_manifest.template.yaml` into a scene folder as `scene_manifest.yaml`, complete it, then add sequentially named originals. Do not edit originals in place. Generated degradations and model outputs belong under `outputs/`.
