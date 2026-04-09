# Phenomenon Study Scripts

This folder implements the study in `phenomenon_study_plan.md`:

- **Synthetic demo (A1/A3)**: data generation, oracle grouping, margin/disagreement/density metrics, and selection failure curves.
- **Real-backbone proxy analysis**: MSG-MVC-style proxy pipeline under **with/without RCI** settings.

## Run

```bash
python -m analysis.synthetic_demo --seed 42 --out-dir analysis/results/synthetic
python -m analysis.backbone_demo --dataset BDGP --seeds 0 1 2 --epochs 8 --out-dir analysis/results/backbone
# or run both
python -m analysis.run_phenomenon_study
```

## Outputs

### Synthetic
- `synthetic_sample_metrics.csv`
- `synthetic_group_summary.csv`
- `selection_strategy_curves.csv`
- `selection_strategy_summary.csv`
- `synthetic_groups_scatter.png`
- `margin_disagreement_density_boxplots.png`
- `selection_precision_contamination.png`

### Backbone
- `proxy_sample_metrics.csv`
- `proxy_group_summary_by_seed.csv`
- `proxy_group_summary_mean_std.csv`
- `proxy_group_distribution.png`
- `flip_rate_by_group.png`
- `loss_or_grad_by_group.png`
- `with_without_rci_comparison.png`
