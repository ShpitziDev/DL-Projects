# Rules for coding agents

1. Read `docs/paper_notes.md`, `docs/decisions.md`, and `docs/experiment_plan.md` before changing experiments.
2. Never launch training, large downloads, checkpoint loading, or dataset acquisition without explicit approval.
3. Never overwrite results; use a unique run directory.
4. Preserve raw custom inputs. Derivatives belong under outputs.
5. Record the resolved configuration and environment metadata for every run.
6. Resolve paths relative to the repository root; do not add user-specific absolute paths.
7. Do not edit code under `external/`; wrap it in `src/vggt_seminar/`.
8. Do not claim successful reproduction without measurements.
9. Clearly distinguish author-reported results, code-verified facts, local observations, assumptions, and hypotheses.
10. Keep notebooks thin; reusable logic belongs in `src/`.
11. Update documentation after meaningful decisions.
12. Do not commit secrets, checkpoints, datasets, or generated geometry.
