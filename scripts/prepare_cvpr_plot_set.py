from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "results" / "plots"
TARGET_DIR = PLOTS_DIR / "k_comparison_cvpr"


MAPPINGS = {
    PLOTS_DIR / "k_comparison" / "honest_01_cv_bars.png": TARGET_DIR / "01_cv_bars.png",
    PLOTS_DIR / "honest_02_noise_curve.png": TARGET_DIR / "02_noise_curve.png",
    PLOTS_DIR / "honest_03_aug_heatmap.png": TARGET_DIR / "03_aug_heatmap.png",
    PLOTS_DIR / "honest_04_roc_pr.png": TARGET_DIR / "04_roc_pr.png",
    PLOTS_DIR / "honest_05_feature_map.png": TARGET_DIR / "05_feature_map.png",
    PLOTS_DIR / "honest_06_summary_table.png": TARGET_DIR / "06_summary_table.png",
    PLOTS_DIR / "repro_02_cv_stability_boxplot.png": TARGET_DIR / "07_cv_stability_boxplot.png",
    PLOTS_DIR / "repro_04_qubit_scaling_bars.png": TARGET_DIR / "08_qubit_scaling_bars.png",
    PLOTS_DIR / "repro_05_pca_manifolds.png": TARGET_DIR / "09_pca_manifolds.png",
    PLOTS_DIR / "repro_06_radar_honest.png": TARGET_DIR / "10_radar.png",
}


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in MAPPINGS.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing source plot: {src}")
        shutil.copy2(src, dst)
    print(str(TARGET_DIR.resolve()))


if __name__ == "__main__":
    main()
