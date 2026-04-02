import cv2
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path

def resize_and_pad(img, target_size=(1920, 1080), pad_color=(255, 255, 255)):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw/w, th/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    top = (th - new_h) // 2
    bottom = th - new_h - top
    left = (tw - new_w) // 2
    right = tw - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def main():
    root_dir = Path("/Volumes/T9/QuantumFeatureSelection")
    demo_dir = root_dir / "cvpr demo"
    video_dir = demo_dir / "video"
    frames_dir = video_dir / "frames"
    
    video_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    
    # Grab all valid visual files and EXPLICITLY ignore all macOS tracking files
    images = [f for f in sorted(list(demo_dir.glob("*.png"))) if not f.name.startswith("._")]
    if not images:
        print("No images found in cvpr demo!")
        return
        
    target_size = (1920, 1080)
    print(f"Generating padded 1080p frames for {len(images)} slides...")
    
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        padded = resize_and_pad(img, target_size)
        cv2.imwrite(str(frames_dir / f"{idx:03d}.png"), padded)
            
    out_path = video_dir / "cvpr_demo_presentation.mp4"
    if out_path.exists():
        out_path.unlink()
        
    print("Stitching slides continuously using FFmpeg H.264 cross-platform codec...")
    cmd = [
        "ffmpeg", "-y", 
        "-framerate", "1/2",           # Display each image for 2 seconds natively
        "-i", str(frames_dir / "%03d.png"),
        "-c:v", "libx264",             # Strict MacOS QuickTime universal support
        "-r", "30",                    # Output 30 FPS container rate
        "-pix_fmt", "yuv420p",         # Standard 4:2:0 subsampling
        str(out_path)
    ]
    subprocess.run(cmd, check=True)
    
    # Cleanup interim padding arrays
    shutil.rmtree(frames_dir)
    
    print(f"Video packaged successfully! Full playback restored: {out_path.name}")

if __name__ == '__main__':
    main()
