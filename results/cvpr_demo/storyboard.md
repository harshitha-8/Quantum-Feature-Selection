# CVPR Demo: Quantum-Enhanced Cotton Defoliation Analytics

A CVPR-style visual story that starts with normal pre- and post-defoliation UAV imagery, adds heatmap-based interpretation, and then walks through the ten core k-comparison evaluation plots with concise captions.

## Slide 1: Pre-Defoliation Normal Image

![Pre-Defoliation Normal Image](../results/cvpr_demo_assets/pre_normal.png)

This opening slide shows a normal pre-defoliation UAV image. The canopy is dense and green, which makes cotton structures harder to isolate directly from raw RGB alone.

## Slide 2: Pre-Defoliation Heatmap

![Pre-Defoliation Heatmap](../results/cvpr_demo_assets/pre_heatmap.png)

The pre-defoliation heatmap highlights the strongest vegetation-response regions. For the demo, this helps viewers understand where the model sees canopy activity before defoliation.

## Slide 3: Post-Defoliation Normal Image

![Post-Defoliation Normal Image](../results/cvpr_demo_assets/post_normal.png)

This normal post-defoliation UAV image shows a drier field appearance with more exposed cotton and reduced green canopy, making the field state visually different from the pre-defoliation view.

## Slide 4: Post-Defoliation Heatmap

![Post-Defoliation Heatmap](../results/cvpr_demo_assets/post_heatmap.png)

The post-defoliation heatmap emphasizes low-vegetation and defoliated field regions. In the CVPR demo, this slide creates a clear visual contrast with the pre-defoliation heatmap.

## Slide 5: k-Comparison CV Bars

![k-Comparison CV Bars](../results/plots/k_comparison/honest_01_cv_bars.png)

This chart compares cross-validation performance across the candidate subsets and gives the first quantitative checkpoint after the qualitative pre/post image slides.

## Slide 6: k-Comparison Noise Curve

![k-Comparison Noise Curve](../results/plots/k_comparison/honest_02_noise_curve.png)

This noise-robustness curve shows how each subset behaves under stronger corruption, which is more informative than clean accuracy for a research demo.

## Slide 7: k-Comparison Augmentation Heatmap

![k-Comparison Augmentation Heatmap](../results/plots/k_comparison/honest_03_aug_heatmap.png)

The augmentation heatmap summarizes performance under fog, glare, and shadow perturbations and works well as a compact robustness slide for CVPR-style presentation.

## Slide 8: k-Comparison ROC and PR

![k-Comparison ROC and PR](../results/plots/k_comparison/honest_04_roc_pr.png)

ROC and precision-recall curves show the ranking quality of the selected subsets and help the audience compare threshold behavior across methods.

## Slide 9: k-Comparison Feature Map

![k-Comparison Feature Map](../results/plots/k_comparison/honest_05_feature_map.png)

This feature-membership map makes it easy to explain which descriptors are included in each subset without interrupting the demo with a dense table.

## Slide 10: k-Comparison Summary Table

![k-Comparison Summary Table](../results/plots/k_comparison/honest_06_summary_table.png)

The summary table condenses the major results into one visual checkpoint and works well as a pause slide in the middle of the demo.

## Slide 11: k-Comparison CV Stability

![k-Comparison CV Stability](../results/plots/k_comparison/repro_02_cv_stability_boxplot.png)

This stability plot shows how consistently the subsets perform across folds, adding an important reliability view to the demo sequence.

## Slide 12: k-Comparison Qubit Scaling

![k-Comparison Qubit Scaling](../results/plots/k_comparison/repro_04_qubit_scaling_bars.png)

The qubit-scaling bars connect subset size to performance trends and are one of the clearest figures for explaining the quantum-design motivation.

## Slide 13: k-Comparison PCA Manifolds

![k-Comparison PCA Manifolds](../results/plots/k_comparison/repro_05_pca_manifolds.png)

The PCA manifold view helps the audience see whether the feature subsets produce meaningful class geometry in a low-dimensional space.

## Slide 14: k-Comparison Robustness Radar

![k-Comparison Robustness Radar](../results/plots/k_comparison/repro_06_radar_honest.png)

The final radar plot gives a compact summary across multiple metrics and works well as the closing quantitative slide in the CVPR demo.
