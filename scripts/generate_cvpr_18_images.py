import os
import shutil
from pathlib import Path
import sys

# Add scripts directory to path to import prepare_cvpr_demo_assets
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import prepare_cvpr_demo_assets as cvpr

OUTPUT_DIR = Path('/Volumes/T9/QuantumFeatureSelection/cvpr demo')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Override CVPR demo assets dir in the imported module so composite functions write to the right folder
cvpr.ASSET_DIR = OUTPUT_DIR 

def main():
    print("Generating pre-defoliation images...")
    # 1. Original
    cvpr.save_single_panel(
        cvpr.PRE_IMAGE,
        "Pre-Defoliation",
        "Original UAV image",
        "01_pre_original.png"
    )
    # 2. Heatmap
    cvpr.save_heat_panel(
        cvpr.PRE_IMAGE,
        "Pre_Defoliation",
        "Pre-Defoliation",
        "Cotton Response Map",
        "02_pre_heatmap.png"
    )
    # 3. Bounding box / Detection overlay
    img = cvpr.open_full_image(cvpr.PRE_IMAGE)
    _, _, detection_view, _ = cvpr.run_cotton_visual_pipeline(img, "Pre_Defoliation")
    panel = cvpr.contain_on_canvas(detection_view, cvpr.CANVAS_SIZE)
    cvpr.add_academic_header(panel, "Pre-Defoliation", "Detected Cotton Regions").save(OUTPUT_DIR / "03_pre_bounding_box.png", format="PNG", optimize=True)

    # 4. Composite (original + heatmap + detection)
    cvpr.build_scene_analysis_figure(
        cvpr.PRE_IMAGE,
        "Pre_Defoliation",
        "Pre-Defoliation Composite",
        "Original, Heatmap, and Bounding Boxes combined",
        "04_pre_composite.png"
    )

    print("Generating post-defoliation images...")
    # 5. Original
    cvpr.save_single_panel(
        cvpr.POST_IMAGE,
        "Post-Defoliation",
        "Original UAV image",
        "05_post_original.png"
    )
    # 6. Heatmap
    cvpr.save_heat_panel(
        cvpr.POST_IMAGE,
        "Post_Defoliation",
        "Post-Defoliation",
        "Cotton Response Map",
        "06_post_heatmap.png"
    )
    
    # 7. Bounding box / Detection overlay
    img_post = cvpr.open_full_image(cvpr.POST_IMAGE)
    _, _, detection_view_post, _ = cvpr.run_cotton_visual_pipeline(img_post, "Post_Defoliation")
    panel_post = cvpr.contain_on_canvas(detection_view_post, cvpr.CANVAS_SIZE)
    cvpr.add_academic_header(panel_post, "Post-Defoliation", "Detected Cotton Regions").save(OUTPUT_DIR / "07_post_bounding_box.png", format="PNG", optimize=True)

    # 8. Composite (original + heatmap + detection)
    cvpr.build_scene_analysis_figure(
        cvpr.POST_IMAGE,
        "Post_Defoliation",
        "Post-Defoliation Composite",
        "Original, Heatmap, and Bounding Boxes combined",
        "08_post_composite.png"
    )
    
    # 9-18. Copy 10 k_comparison plots
    print("Copying k_comparison plots...")
    k_comp_dir = Path("/Volumes/T9/QuantumFeatureSelection/results/plots/k_comparison")
    for file_path in sorted(k_comp_dir.glob("*.png")):
        dest_path = OUTPUT_DIR / file_path.name
        print(f"Copying {file_path.name}")
        shutil.copy2(file_path, dest_path)
    
    print(f"Done! All images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
