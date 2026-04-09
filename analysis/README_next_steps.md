# Next-step Phenomenon Study (v2)

This iteration follows `next_steps_for_codex.md` and focuses on:

1. Fixing disagreement interpretation (`JS(q1, q2)` only).
2. Regenerating pseudo-hard groups with two variants:
   - `D1_outlier_pseudo_hard`
   - `D2_view_corrupted_pseudo_hard`
3. Increasing boundary/conflict sample counts.
4. Re-running group stats, selection curves, and masking study.

## Scripts

- `analysis/regen_synthetic_groups.py`
- `analysis/debug_disagreement.py`
- `analysis/analyze_group_stats.py`
- `analysis/analyze_selection_curves.py`
- `analysis/masking_study.py`
- shared core: `analysis/synth_pipeline.py`

## Suggested run order

```bash
python -m analysis.regen_synthetic_groups --seed 42
python -m analysis.debug_disagreement --seed 42
python -m analysis.analyze_group_stats
python -m analysis.analyze_selection_curves
python -m analysis.masking_study --mask-ratio 0.2 --seeds 0 1 2
```

## Generated outputs

Under `analysis/results/synthetic_v2/`:

- `synthetic_dataset_v2.npz`
- `synthetic_metrics_v2.csv`
- `synthetic_group_counts_v2.csv`
- `synthetic_checklist_v2.csv`
- `debug_disagreement_sanity.csv`
- `debug_disagreement_raw_cases.csv`
- `debug_disagreement_group_summary.csv`
- `group_wise_summary_v2.csv`
- `margin_disagreement_density_boxplots_v2.png`
- `selection_curves_v2.csv`
- `selection_summary_v2.csv`
- `selection_precision_recall_contamination_v2.png`
- `masking_study_results_v2.csv`
- `masking_study_summary_v2.csv`
- `masking_drop_barplot_v2.png`

## Current positioning

- Main narrative: `naive` vs `bc_aware`.
- `trusted_exploratory` is included as an exploratory baseline only.
