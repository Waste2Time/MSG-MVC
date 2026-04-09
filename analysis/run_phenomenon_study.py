from pathlib import Path

from analysis.backbone_demo import BackboneConfig, run_real_backbone_proxy
from analysis.synthetic_demo import SynthConfig, run_a1_a3


def main() -> None:
    root = Path("analysis/results")
    run_a1_a3(SynthConfig(seed=42), root / "synthetic")
    run_real_backbone_proxy(BackboneConfig(dataset="BDGP", seeds=(0, 1, 2), epochs=8), root / "backbone")


if __name__ == "__main__":
    main()
